from unittest import TestCase

from testfixtures import TempDirectory

from elias.fs import extract_file_numbering


class FileSystemUtilsTest(TestCase):

    def test_extract_file_numbering(self):
        with TempDirectory() as d:
            d.makedir("1-apple")
            d.makedir("-5-brussels-sprouts")
            d.makedir("2-banana")
            d.makedir("13-bread")
            d.makedir("4-orange")

            file_numbering = extract_file_numbering(d.path, r"(-?\d+)-.*")

        self.assertEqual(len(file_numbering), 5)

        self.assertEqual(file_numbering[0][0], -5)
        self.assertEqual(file_numbering[0][1], "-5-brussels-sprouts")

        self.assertEqual(file_numbering[1][0], 1)
        self.assertEqual(file_numbering[1][1], "1-apple")

        self.assertEqual(file_numbering[2][0], 2)
        self.assertEqual(file_numbering[2][1], "2-banana")

        self.assertEqual(file_numbering[3][0], 4)
        self.assertEqual(file_numbering[3][1], "4-orange")

        self.assertEqual(file_numbering[4][0], 13)
        self.assertEqual(file_numbering[4][1], "13-bread")