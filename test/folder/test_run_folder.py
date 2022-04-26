from unittest import TestCase

from testfixtures import TempDirectory

from elias.folder import RunFolder
from elias.manager.run import RunManager

TMP_FOLDER: str = None


class TestRunManagerByName(RunManager):

    def __init__(self, run_name: str):
        super(TestRunManagerByName, self).__init__(TMP_FOLDER, run_name)


class TestRunFolder(RunFolder[TestRunManagerByName]):

    def __init__(self):
        super(TestRunFolder, self).__init__(TMP_FOLDER, 'TEST-$[-*]', localize_via_run_name=True)


class RunFolderTest(TestCase):

    def test_resolve_run_name(self):
        with TempDirectory() as d:
            d.makedir("TEST-1")
            d.makedir("TEST-23-name")
            d.makedir("TEST-24-name-with-1-number")

            global TMP_FOLDER
            TMP_FOLDER = d.path

            run_folder = TestRunFolder()
            run_ids = run_folder.list_run_ids()
            self.assertListEqual(run_ids, [1, 23, 24])

            run_name = run_folder.resolve_run_name('TEST-24')
            self.assertEqual(run_name, "TEST-24-name-with-1-number")

            run_manager = run_folder.open_run('TEST-24')
            self.assertEqual(run_manager.get_run_name(), "TEST-24-name-with-1-number")
