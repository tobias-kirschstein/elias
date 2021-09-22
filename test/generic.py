from typing import TypeVar, Generic, Iterable, Iterator, T_co, Container, Collection, List, Deque, Set, Dict, Generator, \
    T, KT, VT, T_contra, V_co
from unittest import TestCase

from elias.generic import get_type_var_instantiation, get_origin, get_type_var_instantiations

_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')


# -------------------------------------------------------------------------
# Dummy Value Classes
# -------------------------------------------------------------------------

class Value1:
    pass


class Value2:
    pass


class Value3:
    pass


# -------------------------------------------------------------------------
# Generic Dummy classes
# -------------------------------------------------------------------------

class SuperClass1TypeVar(Generic[_T1]):
    pass


class SuperClass1TypeVarA(Generic[_T1]):
    pass


class SuperClass1TypeVarB(Generic[_T2]):
    pass


class SuperClass2TypeVar(Generic[_T1, _T2]):
    pass


class MiddleClass1TypeVar(SuperClass1TypeVar[_T1]):
    pass


class MiddleClass1TypeVarA(SuperClass1TypeVarA[_T1]):
    pass


class MiddleClass1TypeVarB(SuperClass1TypeVarA[_T2]):
    pass


class MiddleClass2TypeVar(SuperClass2TypeVar[_T1, _T2]):
    pass


# -------------------------------------------------------------------------
# 1 TypeVar
# -------------------------------------------------------------------------

class TypeVar1(SuperClass1TypeVar[Value1]):
    pass


class TypeVar1Level2(MiddleClass1TypeVar[Value1]):
    pass


# -------------------------------------------------------------------------
# 2 TypeVars
# -------------------------------------------------------------------------

class TypeVar2(SuperClass2TypeVar[Value1, Value2]):
    pass


class TypeVar2Level2(MiddleClass2TypeVar[Value1, Value2]):
    pass


# -------------------------------------------------------------------------
# 2 TypeVars, partial TypeVar instantiation
# -------------------------------------------------------------------------

class MiddleClass2TypeVarInstantiatedFirst(SuperClass2TypeVar[Value1, _T2]):
    pass


class MiddleClass2TypeVarInstantiatedSecond(SuperClass2TypeVar[_T1, Value2]):
    pass


class TypeVar2Level2InstantiatedFirst(MiddleClass2TypeVarInstantiatedFirst[Value2]):
    pass


class TypeVar2Level2InstantiatedSecond(MiddleClass2TypeVarInstantiatedSecond[Value1]):
    pass


# -------------------------------------------------------------------------
# Multiple inheritance, 1 Level
# -------------------------------------------------------------------------

class TypeVar1Super11(SuperClass1TypeVar[Value1], SuperClass1TypeVarB[Value2]):
    pass


class TypeVar2Super12(SuperClass1TypeVar[Value1], SuperClass2TypeVar[Value1, Value2]):
    pass


class TypeVar2Super21(SuperClass2TypeVar[Value1, Value2], SuperClass1TypeVar[Value1]):
    pass


# -------------------------------------------------------------------------
# Multiple inheritance, 2 Levels
# -------------------------------------------------------------------------

class TypeVar2Super11Level2(MiddleClass1TypeVar[Value1], MiddleClass1TypeVarB[Value2]):
    pass


class TypeVar2Super12Level2(MiddleClass1TypeVarA[Value1], MiddleClass2TypeVar[Value1, Value2]):
    pass


class TypeVar2Super21Level2(MiddleClass2TypeVar[Value1, Value2], MiddleClass1TypeVarA[Value1]):
    pass


# -------------------------------------------------------------------------
# Multiple inheritance, partially instantiated
# -------------------------------------------------------------------------

class TypeVar2SuperInstantiatedFirst1(MiddleClass2TypeVarInstantiatedFirst[Value2], SuperClass1TypeVarA[Value1]):
    pass


class TypeVar2SuperInstantiatedSecond1(MiddleClass2TypeVarInstantiatedSecond[Value1], SuperClass1TypeVarA[Value1]):
    pass


class TypeVar2SuperInstantiatedSecondInstantiatedSecond(MiddleClass2TypeVarInstantiatedSecond[Value1],
                                                        MiddleClass2TypeVarInstantiatedFirst[Value2]):
    pass


# -------------------------------------------------------------------------
# Iterable
# -------------------------------------------------------------------------
class SuperClass1Iterable(Iterable[_T1]):

    def __iter__(self) -> Iterator[_T1]:
        pass


class TypeVar1IterableDirect(Iterable[Value1]):

    def __iter__(self) -> Iterator[T_co]:
        pass


class TypeVar1Iterable(SuperClass1Iterable[Value1]):
    pass


class MiddleClass1Iterable(SuperClass1Iterable[_T1]):
    pass


class TypeVar1IterableLevel2(MiddleClass1Iterable[Value1]):
    pass


# -------------------------------------------------------------------------
# Internal Types
# -------------------------------------------------------------------------

class SuperClassIterator(Iterator[_T1]):
    pass


class SuperClassContainer(Container[_T1]):
    pass


class SuperClassCollection(Collection[_T1]):
    pass


class SuperClassList(List[_T1]):
    pass


class SuperClassDeque(Deque[_T1]):
    pass


class SuperClassSet(Set[_T1]):
    pass


class SuperClassDict(Dict[_T1, _T2]):
    pass


class SuperClassGenerator(Generator[_T1, _T2, _T3]):
    pass


class GenericTest(TestCase):

    def test_get_type_var_instantiation(self):
        self.assertEqual(get_type_var_instantiation(SuperClass1TypeVar[Value1], _T1), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClass2TypeVar[Value1, Value2], _T1), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClass2TypeVar[Value1, Value2], _T2), Value2)

        self.assertEqual(get_type_var_instantiation(TypeVar1, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar1Level2, _T1), Value1)

        self.assertEqual(get_type_var_instantiation(TypeVar2, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2, _T2), Value2)
        self.assertEqual(get_type_var_instantiation(TypeVar2Level2, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Level2, _T2), Value2)

        self.assertEqual(get_type_var_instantiation(TypeVar2Level2InstantiatedFirst, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Level2InstantiatedFirst, _T2), Value2)
        self.assertEqual(get_type_var_instantiation(TypeVar2Level2InstantiatedSecond, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Level2InstantiatedSecond, _T2), Value2)

        self.assertEqual(get_type_var_instantiation(TypeVar1Super11, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super12, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super12, _T2), Value2)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super21, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super21, _T2), Value2)

        self.assertEqual(get_type_var_instantiation(TypeVar2Super11Level2, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super11Level2, _T2), Value2)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super12Level2, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super12Level2, _T2), Value2)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super21Level2, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2Super21Level2, _T2), Value2)

        self.assertEqual(get_type_var_instantiation(TypeVar2SuperInstantiatedFirst1, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2SuperInstantiatedFirst1, _T2), Value2)
        self.assertEqual(get_type_var_instantiation(TypeVar2SuperInstantiatedSecond1, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2SuperInstantiatedSecond1, _T2), Value2)
        self.assertEqual(get_type_var_instantiation(TypeVar2SuperInstantiatedSecondInstantiatedSecond, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar2SuperInstantiatedSecondInstantiatedSecond, _T2), Value2)

    def test_get_type_var_instantiation_iterable(self):
        self.assertEqual(get_type_var_instantiation(Iterable[Value1], T_co), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar1IterableDirect, T_co), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar1Iterable, _T1), Value1)
        self.assertEqual(get_type_var_instantiation(TypeVar1IterableLevel2, _T1), Value1)

    def test_get_type_var_instantiation_internal_collections(self):
        self.assertEqual(get_type_var_instantiation(Iterator[Value1], T_co), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassIterator[Value1], _T1), Value1)

        self.assertEqual(get_type_var_instantiation(Container[Value1], T_co), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassContainer[Value1], _T1), Value1)

        self.assertEqual(get_type_var_instantiation(Collection[Value1], T_co), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassCollection[Value1], _T1), Value1)

        self.assertEqual(get_type_var_instantiation(List[Value1], T), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassList[Value1], _T1), Value1)

        self.assertEqual(get_type_var_instantiation(Deque[Value1], T), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassDeque[Value1], _T1), Value1)

        self.assertEqual(get_type_var_instantiation(Set[Value1], T), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassSet[Value1], _T1), Value1)

        self.assertEqual(get_type_var_instantiation(Dict[Value1, Value2], KT), Value1)
        self.assertEqual(get_type_var_instantiation(Dict[Value1, Value2], VT), Value2)
        self.assertEqual(get_type_var_instantiation(SuperClassDict[Value1, Value2], _T1), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassDict[Value1, Value2], _T2), Value2)

        self.assertEqual(get_type_var_instantiation(Generator[Value1, Value2, Value3], T_co), Value1)
        self.assertEqual(get_type_var_instantiation(Generator[Value1, Value2, Value3], T_contra), Value2)
        self.assertEqual(get_type_var_instantiation(Generator[Value1, Value2, Value3], V_co), Value3)
        self.assertEqual(get_type_var_instantiation(SuperClassGenerator[Value1, Value2, Value3], _T1), Value1)
        self.assertEqual(get_type_var_instantiation(SuperClassGenerator[Value1, Value2, Value3], _T2), Value2)
        self.assertEqual(get_type_var_instantiation(SuperClassGenerator[Value1, Value2, Value3], _T3), Value3)
