from typing import List
from unittest import TestCase

from testfixtures import TempDirectory

from elias.folder import Folder


class FolderTest(TestCase):

    def setUp(self) -> None:
        d = TempDirectory()
        d.makedir("1-apple")
        d.makedir("-5-brussels-sprouts")
        d.makedir("2-banana")
        d.makedir("13-bread")
        d.makedir("4-orange")

        d.makedir("TEST-1")
        d.makedir("TEST-23-name")
        d.makedir("TEST-24-name-with-1-number")
        d.makedir("TEST--1")
        d.makedir("TEST--23-name")
        d.makedir("TEST--24-name-with-1-number")

        d.makedir("P2P-9")
        d.makedir("P2P-10")
        d.makedir("P2P-10-12")
        d.makedir("analysis-batch-norm-100-lambda-10")
        d.makedir("analysis-batch-norm-50-lambda-9")
        # Create files
        d.write("epoch-11.ckpt", b'')
        d.write("epoch--1.ckpt", b'')

        self._directory = d
        self._folder = Folder(self._directory.path)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_number_only(self):
        name_format = 'P2P-$'
        expected_result = [(9, 'P2P-9'), (10, 'P2P-10')]

        self._assert_file_numbering_matches(name_format, expected_result)
        self._assert_new_run_name_matches(name_format, 'P2P-11')

    def test_number_in_between(self):
        name_format = 'epoch-$.ckpt'
        expected_result = [(-1, 'epoch--1.ckpt'), (11, 'epoch-11.ckpt')]

        self._assert_file_numbering_matches(name_format, expected_result)
        self._assert_new_run_name_matches(name_format, 'epoch-12.ckpt')

    def test_wild_card_and_number(self):
        name_format = '$-*'
        expected_result = [(-5, "-5-brussels-sprouts"),
                           (1, "1-apple"),
                           (2, "2-banana"),
                           (4, "4-orange"),
                           (13, "13-bread")]

        self._assert_file_numbering_matches(name_format, expected_result)
        self._assert_new_run_name_matches(name_format, '14-cucumber', name='cucumber')

    def test_wild_card_number_and_fixed(self):
        name_format = 'analysis-*-$'
        expected_result = [(9, "analysis-batch-norm-50-lambda-9"), (10, "analysis-batch-norm-100-lambda-10")]

        self._assert_file_numbering_matches(name_format, expected_result)
        self._assert_new_run_name_matches(name_format, 'analysis-batch-norm-no-lambda-11', name='batch-norm-no-lambda')

    def test_no_matches(self):
        name_format = '*-empty-$'
        expected_result = []

        self._assert_file_numbering_matches(name_format, expected_result)
        self._assert_new_run_name_matches(name_format, 'test-empty-1', name='test')

    def test_invalid_format(self):
        with self.assertRaises(AssertionError):
            self._folder.list_file_numbering("no-dollar")

        with self.assertRaises(AssertionError):
            # Multiple $
            self._folder.list_file_numbering("$-$")

        with self.assertRaises(AssertionError):
            # No name supplied
            self._folder.generate_next_name("$-*")

        with self.assertRaises(AssertionError):
            # Name supplied but no wildcard
            self._folder.generate_next_name("P2P-$", name='invalid')

        with self.assertRaises(AssertionError):
            # Multiple wildcards
            self._folder.generate_next_name("*-$-*", name='invalid')

    def test_extract_number(self):
        name_format = 'TEST-$[-*]'
        file_numbering = self._folder.list_file_numbering(name_format)
        file_ids, file_names = zip(*file_numbering)
        self.assertTupleEqual(file_ids, (-24, -23, -1, 1, 23, 24))
        self.assertTupleEqual(file_names,
                              ('TEST--24-name-with-1-number', 'TEST--23-name', 'TEST--1',
                               'TEST-1', 'TEST-23-name', 'TEST-24-name-with-1-number'))

        file_numbering = self._folder.get_numbering_by_file_name(name_format, 'TEST--24-name-with-1-number')
        self.assertEqual(file_numbering, -24)

        file_numbering = self._folder.get_numbering_by_file_name(name_format, 'TEST-1')
        self.assertEqual(file_numbering, 1)

        file_numbering = self._folder.get_numbering_by_file_name(name_format, 'TEST-23')
        self.assertEqual(file_numbering, 23)

        file_name = self._folder.get_file_name_by_numbering(name_format, 1)
        self.assertEqual(file_name, 'TEST-1')

        file_name = self._folder.get_file_name_by_numbering(name_format, -24)
        self.assertEqual(file_name, 'TEST--24-name-with-1-number')

    def _assert_file_numbering_matches(self, name_format: str, expected_result: List):
        expected_file_names = [x[1] for x in expected_result]
        expected_file_numberings = [x[0] for x in expected_result]

        file_names = self._folder.list_file_numbering(name_format, return_only_file_names=True)
        self.assertEqual(file_names, expected_file_names)

        file_numberings = self._folder.list_file_numbering(name_format, return_only_numbering=True)
        self.assertEqual(file_numberings, expected_file_numberings)

        file_numberings_and_names = self._folder.list_file_numbering(name_format)
        self.assertEqual(file_numberings_and_names, expected_result)

    def _assert_new_run_name_matches(self, name_format, expected_name, name=None):
        run_name = self._folder.generate_next_name(name_format, name=name)
        self.assertEqual(run_name, expected_name)
