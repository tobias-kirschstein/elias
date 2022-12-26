from unittest import TestCase

from elias.util.range import IndexRange


class TestIndexRange(TestCase):

    def test_parse_range(self):
        index_range = IndexRange.from_description("5")
        self.assertEqual(index_range.start_idx, 5)
        self.assertEqual(index_range.end_idx, 5)

        index_range = IndexRange.from_description("5-20")
        self.assertEqual(index_range.start_idx, 5)
        self.assertEqual(index_range.end_idx, 20)

        index_range = IndexRange.from_description("5--1")
        index_range.resolve(20)
        self.assertEqual(index_range.start_idx, 5)
        self.assertEqual(index_range.end_idx, 19)

        index_range = IndexRange.from_description("-2")
        index_range.resolve(20)
        self.assertEqual(index_range.start_idx, 18)
        self.assertEqual(index_range.end_idx, 18)

    def test_range_iter(self):
        index_range = IndexRange(5, 19)
        indices = list(index_range)
        self.assertEqual(indices, list(range(5, 20)))

        index_range = IndexRange(5, -1)
        index_range.resolve(100)
        indices = list(index_range)
        self.assertEqual(indices, list(range(5, 100)))
