import os
import pathlib

MILLISECONDS_IN_A_SECOND = 1000
HOURS_IN_A_DAY = 24
DAYS_IN_A_YEAR = 365.25
MILLISECONDS_IN_AN_HOUR = 60 * 60 * MILLISECONDS_IN_A_SECOND
MILLISECONDS_IN_A_DAY = HOURS_IN_A_DAY * MILLISECONDS_IN_AN_HOUR
DAYS_IN_A_MONTH = (DAYS_IN_A_YEAR / 12)


class PathUtility:
    def _get_path_utility_attr(self, attr_name: str, *args: str) -> str:
        """Joins path components and checks if path exists"""
        path = getattr(self, attr_name)
        path = os.path.join(path, *args)
        if not os.path.exists(path):
            print(f'{path} is not a path')
        return path

    def repo_path(self) -> str:
        """Get forecast-benchmarks repo path"""
        if not hasattr(self, 'repo_folder'):
            setattr(self, 'repo_folder',
                    pathlib.Path(__file__).parent.parent.resolve())
        return self._get_path_utility_attr('repo_folder')

    def algo_path(self, algo_name: str, *sub_folders_and_files: str) -> str:
        """Get path within src/algorithms/{algo_name}"""
        if not hasattr(self, algo_name):
            algo_folder_path = os.path.join(self.repo_path(), 'src',
                                            'algorithms', algo_name)
            setattr(self, algo_name, algo_folder_path)
        return self._get_path_utility_attr(algo_name, *sub_folders_and_files)

    def cache_path(self, *sub_folders_and_files: str) -> str:
        """Get path within /cache"""
        if not hasattr(self, 'cache'):
            cache_folder_path = os.path.join(self.repo_path(), 'cache')
            setattr(self, 'cache', cache_folder_path)
        return self._get_path_utility_attr('cache', *sub_folders_and_files)

    def results_path(self, *sub_folders_and_files: str) -> str:
        """Get path within /results"""
        if not hasattr(self, 'results'):
            results_folder_path = os.path.join(self.repo_path(), 'results')
            setattr(self, 'results', results_folder_path)
        return self._get_path_utility_attr('results', *sub_folders_and_files)


PATHS = PathUtility()
