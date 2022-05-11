from typing import List, Union


class Version(object):
    """
    Models a multi-level version similar to SemVer. A version is comprised of numbers separated by dots with an optional
    'v' letter in front.
    Examples: v1.0, 0.0.1, 1.22.333.4444

    The numbers between the dots are referred to as 'levels'. I.e., for v1.0 get_level(0) == 1 and get_level(1) == 0.
    Versions can be compared to their string representations and sorted.
    """

    _levels: List[int]

    def __init__(self, *version_str_or_ints: Union[str, int]):
        if len(version_str_or_ints) == 1 and isinstance(version_str_or_ints[0], str):
            self._levels = Version.parse(version_str_or_ints[0])
        elif len(version_str_or_ints) >= 1 and all([isinstance(part, int) for part in version_str_or_ints]):
            self._levels = list(version_str_or_ints)
        else:
            raise ValueError(f"Version specifier has to a single string or several ints. Got {version_str_or_ints}")

    @staticmethod
    def from_zero(n_levels: int):
        """
        Creates a initial "0" version with the specified amount of levels.
        E.g., 0.0.0

        Parameters
        ----------
            n_levels: how many levels the "0" version should have
        """

        return Version(".".join(["0" for _ in range(n_levels)]))

    @staticmethod
    def from_one(n_levels: int, bump_level: int = 0):
        """
        Creates an initial "1" version with the specified amount of levels.
        E.g., 1.0 or 0.0.1

        Parameters
        ----------
            n_levels: the number of levels
            bump_level: where the 1 should be
        """

        bump_level = bump_level % n_levels
        levels = ["1" if level == bump_level else "0" for level in range(n_levels)]
        return Version(".".join(levels))

    def get_n_levels(self) -> int:
        return len(self._levels)

    def get_level(self, level: int) -> int:
        assert -self.get_n_levels() <= level < self.get_n_levels(), \
            f"Level of bounds. Got `{level}` but only have {self.get_n_levels()} levels"

        return self._levels[level]

    def get_levels(self) -> List[int]:
        return list(self._levels)

    def bump(self, level: int = 0):
        """
        Increases one level by 1 and reverts all lower levels to 0.
        levels are 0-indexed.

        Examples:
        ---------
        >>> version = Version("1.2.3")
        >>> version.bump(0) -> 2.0.0
        >>> version.bump(1) -> 1.3.0
        >>> version.bump(-1) -> 1.2.4

        Parameters
        ----------
            level: which level should be increased
        """

        version_at_level = self.get_level(level)
        level = level % self.get_n_levels()

        self._levels = [v
                        if l < level
                        else
                        (
                            version_at_level + 1
                            if l == level
                            else 0
                        )

                        for l, v in enumerate(self._levels)]

    # -------------------------------------------------------------------------
    # Comparison utilities
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        return ".".join((str(level) for level in self._levels))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Version):
            return self._levels == other._levels
        elif isinstance(other, str):
            return str(self) == other
        else:
            return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, str):
            other = Version(other)

        assert self.get_n_levels() == other.get_n_levels(), \
            f"Versions differ in number of levels. Got {self.get_n_levels()}" and {other.get_n_levels()}

        # Check version levels step by step
        for level in range(self.get_n_levels()):
            lvl_1 = self.get_level(level)
            lvl_2 = other.get_level(level)
            if lvl_1 < lvl_2:
                return True
            elif lvl_1 > lvl_2:
                return False

        # All levels are equal. This version is not smaller than the other
        return False

    # -------------------------------------------------------------------------
    # Parsing utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def parse(version: str) -> List[int]:
        if version.startswith('v'):
            version = version[1:]
        try:
            levels = [Version._parse_level(level) for level in version.split('.')]
            return levels
        except ValueError:
            raise ValueError(f"`{version}` is not a valid version specifier. "
                             f"Only strings that consists of non-negative numbers separated by points are accepted")

    @staticmethod
    def is_valid(version: str) -> bool:
        try:
            Version.parse(version)
            return True
        except ValueError:
            return False

    @staticmethod
    def _parse_level(level: str) -> int:
        level = int(level)
        if level < 0:
            raise ValueError()
        return level
