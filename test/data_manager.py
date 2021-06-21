import unittest
from collections import defaultdict

from elias.data_manager import RandomAccessDataLoader, CombinedRandomAccessDataLoader, IterableDataLoader, \
    CombinedIterableDataLoader, CombinedIterableStopCriterionAnyEmpty, CombinedIterableStopCriterionSpecificEmpty


class ListRADL(RandomAccessDataLoader):

    def __init__(self, data: list):
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


class ListIDL(IterableDataLoader):

    def __init__(self, data: list):
        self._data = data

    def __iter__(self):
        return iter(self._data)


class DataManagerTest(unittest.TestCase):

    def test_combined_random_access_data_loader(self):
        dl_1 = ListRADL(list(range(0, 5)))
        dl_2 = ListRADL(list(range(10, 20)))

        combined_dl = CombinedRandomAccessDataLoader([dl_1, dl_2])
        self.assertEqual(len(combined_dl), 15)
        self.assertEqual(combined_dl[0], (0, 0))
        self.assertEqual(combined_dl[14], (1, 19))
        self.assertEqual(combined_dl[-2], (1, 18))
        self.assertEqual(combined_dl[-10], (1, 10))

        # First 5 should come from dl_1, second 10 from dl_2
        for i, (identifier, _) in enumerate(combined_dl):
            if i < 5:
                self.assertEqual(identifier, 0)
            else:
                self.assertEqual(identifier, 1)

        combined_dl = CombinedRandomAccessDataLoader([dl_1, dl_2],  shuffle=True)
        # Ensure that negative indexing works as expected
        self.assertEqual(combined_dl[0], combined_dl[-15])
        self.assertEqual(combined_dl[2], combined_dl[-13])
        self.assertEqual(combined_dl[10], combined_dl[-5])

        # Ensure that we can loop through the data loader a second time
        for _ in combined_dl:
            pass

    def test_combined_iterable_data_loader(self):
        dl_1 = ListIDL(list(range(0, 5)))
        dl_2 = ListIDL(list(range(10, 20)))

        combined_dl = CombinedIterableDataLoader([dl_1, dl_2])
        # First 5 should come from dl_1, second 10 from dl_2
        i = 0
        for i, (identifier, value) in enumerate(combined_dl):
            if i < 5:
                self.assertEqual(identifier, 0)
                self.assertEqual(value, i)
            else:
                self.assertEqual(identifier, 1)
                self.assertEqual(value, i + 5)
        self.assertEqual(i, 14)  # After looping through the data loader we should have 15 elements

        combined_dl = CombinedIterableDataLoader([dl_1, dl_2], shuffle=True)
        elements = set()
        identifier_count = defaultdict(lambda: 0)
        for identifier, value in combined_dl:
            elements.add(value)
            identifier_count[identifier] += 1

        # Check that we have seen each element exactly once
        self.assertEqual(len(elements), 15)
        self.assertEqual(identifier_count[0], 5)
        self.assertEqual(identifier_count[1], 10)

        # Ensure that we can loop through the data loader a second time
        for _ in combined_dl:
            pass

    def test_combined_iterable_stop_criterion(self):
        dl_1 = ListIDL(list(range(0, 5)))
        dl_2 = ListIDL(list(range(10, 20)))

        combined_dl = CombinedIterableDataLoader([dl_1, dl_2], stop_criterion=CombinedIterableStopCriterionAnyEmpty())
        i = 0
        for i, (identifier, value) in enumerate(combined_dl):
            self.assertEqual(identifier, 0)
            self.assertEqual(value, i)
        self.assertEqual(i, 4)  # After looping through the data loader we should have 5 elements

        combined_dl = CombinedIterableDataLoader([dl_1, dl_2], stop_criterion=CombinedIterableStopCriterionSpecificEmpty(1))
        i = 0
        for i, (identifier, value) in enumerate(combined_dl):
            if i < 5:
                self.assertEqual(identifier, 0)
                self.assertEqual(value, i)
            else:
                self.assertEqual(identifier, 1)
                self.assertEqual(value, i + 5)
        self.assertEqual(i, 14)  # After looping through the data loader we should have 5 elements

        combined_dl = CombinedIterableDataLoader([dl_1, dl_2],
                                                 stop_criterion=CombinedIterableStopCriterionSpecificEmpty(1),
                                                 shuffle=True)
        elements = set()
        identifier_count = defaultdict(lambda: 0)
        for identifier, value in combined_dl:
            elements.add(value)
            identifier_count[identifier] += 1

        # Check that we have at least seen all elements of the second dataloader
        self.assertEqual(identifier_count[1], 10)

        combined_dl = CombinedIterableDataLoader([dl_1, dl_2],
                                                 stop_criterion=CombinedIterableStopCriterionAnyEmpty(),
                                                 shuffle=True,
                                                 sample_weights=[0, 1])

        elements = set()
        identifier_count = defaultdict(lambda: 0)
        for identifier, value in combined_dl:
            elements.add(value)
            identifier_count[identifier] += 1

        # Check that we have at least seen all elements of the second dataloader
        self.assertEqual(identifier_count[1], 10)
        self.assertEqual(identifier_count[0], 0)

        dl_1 = ListIDL(list(range(0, 1000)))
        dl_2 = ListIDL(list(range(1000, 2000)))

        combined_dl = CombinedIterableDataLoader([dl_1, dl_2],
                                                 stop_criterion=CombinedIterableStopCriterionAnyEmpty(),
                                                 shuffle=True,
                                                 sample_weights=[0.1, 0.9])

        elements = set()
        identifier_count = defaultdict(lambda: 0)
        for identifier, value in combined_dl:
            elements.add(value)
            identifier_count[identifier] += 1

        # Check that we have at least seen all elements of the second dataloader
        self.assertGreater(identifier_count[1], identifier_count[0])
        self.assertGreater(identifier_count[0], 0)
        self.assertLess(identifier_count[0], 300)

        # Ensure that we can loop through the data loader a second time
        for _ in combined_dl:
            pass

