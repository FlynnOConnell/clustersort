import unittest
from pathlib import Path
from spk2py.merge import merge_files


class TestMergeFiles(unittest.TestCase):

    def setUp(self):
        # Setup code: creating some temporary .smr files for testing
        self.test_dir = Path('./test_data')
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.file1 = self.test_dir / 'testfile1_preinfusion.smr'
        self.file2 = self.test_dir / 'testfile2_postinfusion.smr'
        self.file1.touch()
        self.file2.touch()

    def tearDown(self):
        self.file1.unlink()
        self.file2.unlink()
        self.test_dir.rmdir()

    def test_merge_files(self):
        savepath = Path('./test_data_output')
        savepath.mkdir(parents=True, exist_ok=True)

        try:
            merge_files(filepath=self.test_dir, savepath=savepath)
            self.assertTrue((savepath / 'file1_combined.hdf5').is_file())
        finally:
            for file in savepath.iterdir():
                file.unlink()
            savepath.rmdir()


if __name__ == "__main__":
    unittest.main()
