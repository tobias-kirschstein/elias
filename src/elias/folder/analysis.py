from shutil import rmtree
from typing import TypeVar, Generic, Type, Union

from silberstral import reveal_type_var

from elias.manager.analysis import Analysis
from elias.folder.run import RunFolder

_AnalysisType = TypeVar("_AnalysisType", bound=Analysis)


class AnalysisFolder(Generic[_AnalysisType], RunFolder):
    _analysis_cls: Type[_AnalysisType]

    def __init__(self, analysis_folder: str, name_format: str = '$-*'):
        super(AnalysisFolder, self).__init__(analysis_folder, name_format)
        self._analysis_cls = reveal_type_var(self, _AnalysisType)

    def open_analysis(self, analysis_name_or_id: Union[str, int]) -> _AnalysisType:
        analysis_name = self.resolve_run_name(analysis_name_or_id)
        return self._analysis_cls.from_location(f"{self._folder.get_location()}/{analysis_name}")

    def new_analysis(self, analysis_name: str) -> _AnalysisType:
        new_analysis_name = self.generate_run_name(name=analysis_name)
        return self.open_analysis(new_analysis_name)

    def delete_analysis(self, analysis_name_or_id: Union[str, int]):
        analysis_name = self.resolve_run_name(analysis_name_or_id)
        rmtree(f"{self._folder.get_location()}/{analysis_name}")