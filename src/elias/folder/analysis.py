from shutil import rmtree
from typing import TypeVar, Type, Union

from elias.folder.run import RunFolder
from elias.manager.analysis import Analysis

_AnalysisType = TypeVar("_AnalysisType", bound=Analysis)


class AnalysisFolder(RunFolder[_AnalysisType]):

    _cls_run_manager: Type[_AnalysisType]

    def __init__(self, analysis_folder: str, name_format: str = '$-*', localize_via_run_name: bool = False):
        super(AnalysisFolder, self).__init__(analysis_folder, name_format, localize_via_run_name=localize_via_run_name)

    def open_analysis(self, analysis_name_or_id: Union[str, int]) -> _AnalysisType:
        analysis_name = self.resolve_run_name(analysis_name_or_id)
        return self._cls_run_manager.from_location(self._folder.get_location(),
                                                   analysis_name,
                                                   localize_via_analysis_name=self._localize_via_run_name)

    def new_analysis(self, analysis_name: str) -> _AnalysisType:
        new_analysis_name = self.generate_run_name(name=analysis_name)
        return self.open_analysis(new_analysis_name)

    def delete_analysis(self, analysis_name_or_id: Union[str, int]):
        analysis_name = self.resolve_run_name(analysis_name_or_id)
        rmtree(f"{self._folder.get_location()}/{analysis_name}")