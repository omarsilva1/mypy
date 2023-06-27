"""Test cases for mypy types and type operations."""

from __future__ import annotations

from mypy.erasetype import erase_type, remove_instance_last_known_values
from mypy.expandtype import expand_type
from mypy.indirection import TypeIndirectionVisitor
from mypy.join import join_simple, join_types
from mypy.meet import meet_types, narrow_declared_type
from mypy.nodes import ARG_OPT, ARG_POS, ARG_STAR, ARG_STAR2, CONTRAVARIANT, COVARIANT, INVARIANT
from mypy.state import state
from mypy.subtypes import is_more_precise, is_proper_subtype, is_same_type, is_subtype, simplify_omega, \
    convert_to_cnf, convert_to_dnf, convert_to_anf, is_xi_subtype
from mypy.test.helpers import Suite, assert_equal, assert_type, skip
from mypy.test.typefixture import InterfaceTypeFixture, TypeFixture
from mypy.typeops import false_only, make_simplified_union, true_only
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    LiteralType,
    NoneType,
    Overloaded,
    ProperType,
    TupleType,
    Type,
    TypeOfAny,
    TypeType,
    TypeVarId,
    TypeVarType,
    UnboundType,
    UninhabitedType,
    UnionType,
    IntersectionType,
    get_proper_type,
    has_recursive_types,
)


class TypesSuite(Suite):
    def setUp(self) -> None:
        self.x = UnboundType("X")  # Helpers
        self.y = UnboundType("Y")
        self.fx = TypeFixture()
        self.function = self.fx.function

    def test_any(self) -> None:
        assert_equal(str(AnyType(TypeOfAny.special_form)), "Any")

    def test_simple_unbound_type(self) -> None:
        u = UnboundType("Foo")
        assert_equal(str(u), "Foo?")

    def test_generic_unbound_type(self) -> None:
        u = UnboundType("Foo", [UnboundType("T"), AnyType(TypeOfAny.special_form)])
        assert_equal(str(u), "Foo?[T?, Any]")

    def test_callable_type(self) -> None:
        c = CallableType(
            [self.x, self.y],
            [ARG_POS, ARG_POS],
            [None, None],
            AnyType(TypeOfAny.special_form),
            self.function,
        )
        assert_equal(str(c), "def (X?, Y?) -> Any")

        c2 = CallableType([], [], [], NoneType(), self.fx.function)
        assert_equal(str(c2), "def ()")

    def test_callable_type_with_default_args(self) -> None:
        c = CallableType(
            [self.x, self.y],
            [ARG_POS, ARG_OPT],
            [None, None],
            AnyType(TypeOfAny.special_form),
            self.function,
        )
        assert_equal(str(c), "def (X?, Y? =) -> Any")

        c2 = CallableType(
            [self.x, self.y],
            [ARG_OPT, ARG_OPT],
            [None, None],
            AnyType(TypeOfAny.special_form),
            self.function,
        )
        assert_equal(str(c2), "def (X? =, Y? =) -> Any")

    def test_callable_type_with_var_args(self) -> None:
        c = CallableType(
            [self.x], [ARG_STAR], [None], AnyType(TypeOfAny.special_form), self.function
        )
        assert_equal(str(c), "def (*X?) -> Any")

        c2 = CallableType(
            [self.x, self.y],
            [ARG_POS, ARG_STAR],
            [None, None],
            AnyType(TypeOfAny.special_form),
            self.function,
        )
        assert_equal(str(c2), "def (X?, *Y?) -> Any")

        c3 = CallableType(
            [self.x, self.y],
            [ARG_OPT, ARG_STAR],
            [None, None],
            AnyType(TypeOfAny.special_form),
            self.function,
        )
        assert_equal(str(c3), "def (X? =, *Y?) -> Any")

    def test_tuple_type(self) -> None:
        assert_equal(str(TupleType([], self.fx.std_tuple)), "Tuple[]")
        assert_equal(str(TupleType([self.x], self.fx.std_tuple)), "Tuple[X?]")
        assert_equal(
            str(TupleType([self.x, AnyType(TypeOfAny.special_form)], self.fx.std_tuple)),
            "Tuple[X?, Any]",
        )

    def test_type_variable_binding(self) -> None:
        assert_equal(str(TypeVarType("X", "X", 1, [], self.fx.o)), "X`1")
        assert_equal(str(TypeVarType("X", "X", 1, [self.x, self.y], self.fx.o)), "X`1")

    def test_generic_function_type(self) -> None:
        c = CallableType(
            [self.x, self.y],
            [ARG_POS, ARG_POS],
            [None, None],
            self.y,
            self.function,
            name=None,
            variables=[TypeVarType("X", "X", -1, [], self.fx.o)],
        )
        assert_equal(str(c), "def [X] (X?, Y?) -> Y?")

        v = [TypeVarType("Y", "Y", -1, [], self.fx.o), TypeVarType("X", "X", -2, [], self.fx.o)]
        c2 = CallableType([], [], [], NoneType(), self.function, name=None, variables=v)
        assert_equal(str(c2), "def [Y, X] ()")

    def test_type_alias_expand_once(self) -> None:
        A, target = self.fx.def_alias_1(self.fx.a)
        assert get_proper_type(A) == target
        assert get_proper_type(target) == target

        A, target = self.fx.def_alias_2(self.fx.a)
        assert get_proper_type(A) == target
        assert get_proper_type(target) == target

    def test_type_alias_expand_all(self) -> None:
        A, _ = self.fx.def_alias_1(self.fx.a)
        assert A.expand_all_if_possible() is None
        A, _ = self.fx.def_alias_2(self.fx.a)
        assert A.expand_all_if_possible() is None

        B = self.fx.non_rec_alias(self.fx.a)
        C = self.fx.non_rec_alias(TupleType([B, B], Instance(self.fx.std_tuplei, [B])))
        assert C.expand_all_if_possible() == TupleType(
            [self.fx.a, self.fx.a], Instance(self.fx.std_tuplei, [self.fx.a])
        )

    def test_recursive_nested_in_non_recursive(self) -> None:
        A, _ = self.fx.def_alias_1(self.fx.a)
        T = TypeVarType("T", "T", -1, [], self.fx.o)
        NA = self.fx.non_rec_alias(Instance(self.fx.gi, [T]), [T], [A])
        assert not NA.is_recursive
        assert has_recursive_types(NA)

    def test_indirection_no_infinite_recursion(self) -> None:
        A, _ = self.fx.def_alias_1(self.fx.a)
        visitor = TypeIndirectionVisitor()
        modules = A.accept(visitor)
        assert modules == {"__main__", "builtins"}

        A, _ = self.fx.def_alias_2(self.fx.a)
        visitor = TypeIndirectionVisitor()
        modules = A.accept(visitor)
        assert modules == {"__main__", "builtins"}


class TypeOpsSuite(Suite):
    def setUp(self) -> None:
        self.fx = TypeFixture(INVARIANT)
        self.fx_co = TypeFixture(COVARIANT)
        self.fx_contra = TypeFixture(CONTRAVARIANT)

    # expand_type

    def test_trivial_expand(self) -> None:
        for t in (
            self.fx.a,
            self.fx.o,
            self.fx.t,
            self.fx.nonet,
            self.tuple(self.fx.a),
            self.callable([], self.fx.a, self.fx.a),
            self.fx.anyt,
        ):
            self.assert_expand(t, [], t)
            self.assert_expand(t, [], t)
            self.assert_expand(t, [], t)

    def test_trivial_expand_recursive(self) -> None:
        A, _ = self.fx.def_alias_1(self.fx.a)
        self.assert_expand(A, [], A)
        A, _ = self.fx.def_alias_2(self.fx.a)
        self.assert_expand(A, [], A)

    def test_expand_naked_type_var(self) -> None:
        self.assert_expand(self.fx.t, [(self.fx.t.id, self.fx.a)], self.fx.a)
        self.assert_expand(self.fx.t, [(self.fx.s.id, self.fx.a)], self.fx.t)

    def test_expand_basic_generic_types(self) -> None:
        self.assert_expand(self.fx.gt, [(self.fx.t.id, self.fx.a)], self.fx.ga)

    # IDEA: Add test cases for
    #   tuple types
    #   callable types
    #   multiple arguments

    def assert_expand(
        self, orig: Type, map_items: list[tuple[TypeVarId, Type]], result: Type
    ) -> None:
        lower_bounds = {}

        for id, t in map_items:
            lower_bounds[id] = t

        exp = expand_type(orig, lower_bounds)
        # Remove erased tags (asterisks).
        assert_equal(str(exp).replace("*", ""), str(result))

    # erase_type

    def test_trivial_erase(self) -> None:
        for t in (self.fx.a, self.fx.o, self.fx.nonet, self.fx.anyt):
            self.assert_erase(t, t)

    def test_erase_with_type_variable(self) -> None:
        self.assert_erase(self.fx.t, self.fx.anyt)

    def test_erase_with_generic_type(self) -> None:
        self.assert_erase(self.fx.ga, self.fx.gdyn)
        self.assert_erase(self.fx.hab, Instance(self.fx.hi, [self.fx.anyt, self.fx.anyt]))

    def test_erase_with_generic_type_recursive(self) -> None:
        tuple_any = Instance(self.fx.std_tuplei, [AnyType(TypeOfAny.explicit)])
        A, _ = self.fx.def_alias_1(self.fx.a)
        self.assert_erase(A, tuple_any)
        A, _ = self.fx.def_alias_2(self.fx.a)
        self.assert_erase(A, UnionType([self.fx.a, tuple_any]))

    def test_erase_with_tuple_type(self) -> None:
        self.assert_erase(self.tuple(self.fx.a), self.fx.std_tuple)

    def test_erase_with_function_type(self) -> None:
        self.assert_erase(
            self.fx.callable(self.fx.a, self.fx.b),
            CallableType(
                arg_types=[self.fx.anyt, self.fx.anyt],
                arg_kinds=[ARG_STAR, ARG_STAR2],
                arg_names=[None, None],
                ret_type=self.fx.anyt,
                fallback=self.fx.function,
            ),
        )

    def test_erase_with_type_object(self) -> None:
        self.assert_erase(
            self.fx.callable_type(self.fx.a, self.fx.b),
            CallableType(
                arg_types=[self.fx.anyt, self.fx.anyt],
                arg_kinds=[ARG_STAR, ARG_STAR2],
                arg_names=[None, None],
                ret_type=self.fx.anyt,
                fallback=self.fx.type_type,
            ),
        )

    def test_erase_with_type_type(self) -> None:
        self.assert_erase(self.fx.type_a, self.fx.type_a)
        self.assert_erase(self.fx.type_t, self.fx.type_any)

    def assert_erase(self, orig: Type, result: Type) -> None:
        assert_equal(str(erase_type(orig)), str(result))

    # is_more_precise

    def test_is_more_precise(self) -> None:
        fx = self.fx
        assert is_more_precise(fx.b, fx.a)
        assert is_more_precise(fx.b, fx.b)
        assert is_more_precise(fx.b, fx.b)
        assert is_more_precise(fx.b, fx.anyt)
        assert is_more_precise(self.tuple(fx.b, fx.a), self.tuple(fx.b, fx.a))
        assert is_more_precise(self.tuple(fx.b, fx.b), self.tuple(fx.b, fx.a))

        assert not is_more_precise(fx.a, fx.b)
        assert not is_more_precise(fx.anyt, fx.b)

    # is_proper_subtype

    def test_is_proper_subtype(self) -> None:
        fx = self.fx

        assert is_proper_subtype(fx.a, fx.a)
        assert is_proper_subtype(fx.b, fx.a)
        assert is_proper_subtype(fx.b, fx.o)
        assert is_proper_subtype(fx.b, fx.o)

        assert not is_proper_subtype(fx.a, fx.b)
        assert not is_proper_subtype(fx.o, fx.b)

        assert is_proper_subtype(fx.anyt, fx.anyt)
        assert not is_proper_subtype(fx.a, fx.anyt)
        assert not is_proper_subtype(fx.anyt, fx.a)

        assert is_proper_subtype(fx.ga, fx.ga)
        assert is_proper_subtype(fx.gdyn, fx.gdyn)
        assert not is_proper_subtype(fx.ga, fx.gdyn)
        assert not is_proper_subtype(fx.gdyn, fx.ga)

        assert is_proper_subtype(fx.t, fx.t)
        assert not is_proper_subtype(fx.t, fx.s)

        assert is_proper_subtype(fx.a, UnionType([fx.a, fx.b]))
        assert is_proper_subtype(UnionType([fx.a, fx.b]), UnionType([fx.a, fx.b, fx.c]))
        assert not is_proper_subtype(UnionType([fx.a, fx.b]), UnionType([fx.b, fx.c]))

    def test_simplify_omega(self) -> None:
        # ω ∩ σ and σ ∩ ω rewrite to σ;
        # ω ∪ σ and σ ∪ ω rewrite to ω;
        # σ → ω rewrites to ω.
        fx = self.fx_co
        omega = fx.anyt

        test_cases = [
            (IntersectionType([omega, fx.a]), fx.a),
            (IntersectionType([fx.a, omega]), fx.a),
            (IntersectionType([fx.a, fx.b, fx.d, omega]), IntersectionType([fx.a, fx.b, fx.d])),
            (IntersectionType([fx.a, fx.d]), IntersectionType([fx.a, fx.d])),
            (UnionType([omega, fx.a]), omega),
            (UnionType([fx.a, omega]), omega),
            (UnionType([fx.a, fx.b, fx.d, omega]), omega),
            (UnionType([fx.a, fx.b, fx.d]), UnionType([fx.a, fx.b, fx.d])),
            (fx.callable(fx.a, omega), omega),
            (fx.callable(fx.a, fx.d), fx.callable(fx.a, fx.d)),
            (fx.callable(omega, fx.a), fx.callable(omega, fx.a))
        ]

        for test_case, expected_result in test_cases:
            simplified = simplify_omega(test_case)
            assert_equal(simplified, expected_result)

    def test_convert_to_cnf(self) -> None:
        fx = self.fx_co
        def intersect(*a: Type) -> IntersectionType:
            return IntersectionType(list(a))

        def union(*a: Type) -> UnionType:
            return UnionType(list(a))

        def arrow(self, *a: Type) -> CallableType:
            return fx.callable(self, *a)

        # σ ∪ (τ ∩ ρ) rewrites to (σ ∪ τ ) ∩ (σ ∪ ρ);
        # (σ ∩ τ ) ∪ ρ rewrites to (σ ∪ ρ) ∩ (τ ∪ ρ);
        a = fx.a
        b = fx.b
        c = fx.c
        d = fx.d
        e = fx.e

        test_cases = [
            # a ∨ (b ∧ c) => (a ∨ b) ∧ (a ∨ c)
            (union(a, intersect(b, c)), intersect(union(a, b), union(a, c))),
            # (a ∧ b) ∨ c => (a ∨ c) ∧ (b ∨ c)
            (union(intersect(a, b), c), intersect(union(a, c), union(b, c))),
            # (a ∧ b) ∧ (c ∨ (d ∧ e)) => a ∧ b ∧ (c ∨ d) ∧ (c ∨ e)
            (intersect(intersect(a, b), union(c, intersect(d, e))), intersect(a, b, union(c, d), union(c, e))),
            # (a ∧ b) ∨ (c ∧ d) => (a ∨ c) ∧ (a ∨ d) ∧ (b ∨ c) ∧ (b ∨ d)
            (union(intersect(a, b), intersect(c, d)), intersect(
                union(a, c),
                union(a, d),
                union(b, c),
                union(b, d),
            )),
            #
            # (a ∧ b) ∨ (c ∧ d ∧ e) => (a ∨ c) ∧ (a ∨ d) ∧ (a ∨ e) ∧ (b ∨ c) ∧ (b ∨ d) ∧ (b ∨ e)
            (union(intersect(a, b), intersect(c, d, e)), intersect(
                union(a, c),
                union(a, d),
                union(a, e),
                union(b, c),
                union(b, d),
                union(b, e)
            )),
            #  (c ∧ d ∧ e) ∨ (a ∧ b)  => (a ∨ c) ∧ (a ∨ d) ∧ (a ∨ e) ∧ (b ∨ c) ∧ (b ∨ d) ∧ (b ∨ e)
            (union(intersect(c, d, e), intersect(a, b)), intersect(
                union(a, c),
                union(a, d),
                union(a, e),
                union(b, c),
                union(b, d),
                union(b, e)
            )),

            # # more test cases
            # (a ∨ b) ∧ (c ∨ d) => (a ∨ b) ∧ (c ∨ d) (unchanged since already in CNF)
            (intersect(union(a, b), union(c, d)), intersect(
                union(a, b),
                union(c, d),
            )),

            (
                intersect(a, union(b, (intersect(c, d)))),
                intersect(a, union(b, c), union(b, d))
            ),

            # (a ∧ b) ∨ (c ∧ d) => (a ∨ c) ∧ (a ∨ d) ∧ (b ∨ c) ∧ (b ∨ d)
            (union(intersect(a, b), intersect(c, d)), intersect(
                union(a, c),
                union(a, d),
                union(b, c),
                union(b, d),
            )),

            # a ∨ (b ∧ c) => (a ∨ b) ∧ (a ∨ c)
            (union(a, intersect(b, c)), intersect(
                union(a, b),
                union(a, c),
            )),

            # (a ∧ b ∧ c) ∨ d => (a ∨ d) ∧ (b ∨ d) ∧ (c ∨ d)
            (union(intersect(a, b, c), d), intersect(
                union(a, d),
                union(b, d),
                union(c, d),
            )),

            # a ∨ (b ∧ c ∧ d) => (a ∨ b) ∧ (a ∨ c) ∧ (a ∨ d)
            (union(a, intersect(b, c, d)), intersect(
                union(a, b),
                union(a, c),
                union(a, d),
            )),

            # a ∨ b ∨ (c ∧ b ∧ d) => (a ∨ b ∨ c) ∧ (a ∨ b ∨ d) ∧ (a ∨ b ∨ e)
            (union(a, b, intersect(c, d, e)), intersect(union(a, b, c), union(a, b, d), union(a, b, e))),

            # # some with arrow
            # (a -> b) ∨ (c ∧ d) => ((a -> b) ∨ c) ∧ ((a -> b) ∨ d)
            (union(arrow(a, b), intersect(c, d)), intersect(
                union(arrow(a, b), c),
                union(arrow(a, b), d),
            )),
        ]

        for test_case, expected_result in test_cases:
            converted = convert_to_cnf(test_case)
            assert_equal(converted, expected_result)

    def test_convert_to_dnf(self) -> None:
        fx = self.fx_co
        def intersect(*a: Type) -> IntersectionType:
            return IntersectionType(list(a))

        def union(*a: Type) -> UnionType:
            return UnionType(list(a))

        def arrow(self, *a: Type) -> CallableType:
            return fx.callable(self, *a)

        # – σ ∩ (τ ∪ ρ) rewrites to (σ ∩ τ ) ∪ (σ ∩ ρ);
        # – (σ ∪ τ) ∩ ρ rewrites to (σ ∩ ρ) ∪ (τ ∩ ρ).
        a = fx.a
        b = fx.b
        c = fx.c
        d = fx.d
        e = fx.e

        test_cases = [
            # (a ∨ b) => (a ∨ b)
            (union(a, b), union(a, b)),

            # (a ∧ b) => (a ∧ b)
            (intersect(a, b), intersect(a, b)),

            # (a ∧ b) ∨ (c ∧ d) => (a ∧ b) ∨ (c ∧ d)
            (union(intersect(a, b), intersect(c, d)), union(intersect(a, b), intersect(c, d))),

            # a ∧ (c ∨ b) => (a ∧ c) ∨ (a ∧ b)
            (intersect(a, union(c, b)), union(intersect(a, c), intersect(a, b))),

            # (c ∨ b) ∧ a => (a ∧ c) ∨ (a ∧ b)
            (intersect(union(c, b), a), union(intersect(a, c), intersect(a, b))),

            # a ∧ (c ∨ b ∨ d) => (a ∧ b) ∨ (a ∧ c) ∨ (a ∧ d)
            (intersect(a, union(c, b, d)), union(intersect(a, b), intersect(a, c), intersect(a, d))),

            # a ∧ b ∧ (c ∨ b ∨ d) => (a ∧ b ∧ c) ∨ (a ∧ b ∧ d) ∨ (a ∧ b ∧ e)
            (intersect(a, b, union(c, d, e)), union(intersect(a, b, c), intersect(a, b, d), intersect(a, b, e))),

            # (a ∨ (b ∧ (c ∨ d)) => a ∨ (b ∧ c) ∨ (b ∧ d)
            (union(a, intersect(b, union(c, d))), union(a, intersect(c, b), intersect(d, b))),

            # (a -> b) => (a -> b)
            (arrow(a, b), arrow(a, b)),

            # a -> (b ∧ (c ∨ d)) => a -> ((b ∧ c) ∨ (b ∧ d))
            (arrow(a, intersect(b, union(c, d))), arrow(a, union(intersect(b, c), intersect(b, d)))),

            # (b ∧ (c ∨ d)) -> a => ((b ∧ c) ∨ (b ∧ d)) -> a
            (arrow(intersect(b, union(c, d)), a), arrow(union(intersect(b, c), intersect(b, d)), a)),

        ]

        for test_case, expected_result in test_cases:
            converted = convert_to_dnf(test_case)
            assert_equal(converted, expected_result)

    def test_convert_to_anf(self) -> None:
        fx = self.fx_co
        def intersect(*a: Type) -> IntersectionType:
            return IntersectionType(list(a))

        def union(*a: Type) -> UnionType:
            return UnionType(list(a))

        def arrow(self, *a: Type) -> CallableType:
            return fx.callable(self, *a)

        # - σ → τ rewrites to DNF(σ) → CNF(τ);
        # – ∪iσi → ∩j τj rewrites to ∩i(∩j(σi → τj )).
        omega = fx.anyt
        a = fx.a
        b = fx.b
        c = fx.c
        d = fx.d
        e = fx.e
        f = fx.f


        test_cases = [

            # TODO OMAR: new case [(a ∨ b), c, d] -> e => ([a, c, d] -> e) ∧ ([b, c, d] -> e)
            (arrow(union(a, b), c, d, e), intersect(
                arrow(a, c, d, e),
                arrow(b, c, d, e)
            )),

            (intersect(arrow(union(a, b), c, d, e), a), intersect(
                arrow(a, c, d, e),
                arrow(b, c, d, e),
                a
            )),

            (union(arrow(union(a, b), c, d, e), a), union(intersect(
                arrow(a, c, d, e),
                arrow(b, c, d, e)),
                a
            )),

            # ((a ∨ b),(c v d)) -> c => (a,c → e) ∧ (a,d → e) ∧ (b,c → e) ∧ (b,d → e)
            (arrow(union(a, b), union(c, d), e), intersect(
                arrow(a, c, e),
                arrow(a, d, e),
                arrow(b, c, e),
                arrow(b, d, e),
            )),

            # ω => ω
            (omega, omega),

            # a => a
            (a, a),

            # a -> b => a -> b
            (arrow(a, b), arrow(a, b)),

            # (a -> b) -> c => (a -> b) -> c
            (arrow(arrow(a, b), c), arrow(arrow(a, b), c)),

            # ω -> (a ∧ b ∧ c) => ω -> a ∧ ω -> b ∧ ω -> c
            (arrow(omega, intersect(a, b, c)), intersect(
                arrow(omega, a),
                arrow(omega, b),
                arrow(omega, c)
            )),

            # (a ∨ b) -> c => (a → c) ∧ (b → c))
            (arrow(union(a, b), c), intersect(
                arrow(a, c),
                arrow(b, c)
            )),

            # (a -> (b ∧ c)) => (a → b) ∧ (a → c))
            (arrow(a, intersect(b, c)), intersect(
                arrow(a, b),
                arrow(a, c)
            )),

            # (a ∨ (b ∧ (c ∨ d)) -> e => ((a → e) ∧ ((b ∧ c) → e) ∧ ((b ∧ d) → e))
            (arrow(union(a, intersect(b, union(c, d))), e), intersect(
                arrow(a, e),
                arrow(intersect(b, c), e),
                arrow(intersect(b, d), e)
            )),

            # (a ∨ b) -> (c ∧ d) => (a -> c) ∧ (a -> d) ∧ (b -> c) ∧ (b -> d)
            (arrow(union(a, b), intersect(c, d)), intersect(
                arrow(a, c),
                arrow(a, d),
                arrow(b, c),
                arrow(b, d)
            )),

            # (a ∨ b ∨ c) ->  (d ∧ e) => (a -> d) ∧ (a -> e) ∧ (b -> d) ∧ (b -> e) ∧ (c -> d) ∧ (c -> e)
            (arrow(union(a, b, c), intersect(d, e)), intersect(
                arrow(a, d),
                arrow(a, e),
                arrow(b, d),
                arrow(b, e),
                arrow(c, d),
                arrow(c, e)
            )),

            # (a ∨ b) ->  (c ∧ d ∧ e) => (a -> c) ∧ (a -> d) ∧ (a -> e) ∧ (b -> c) ∧ (b -> d) ∧ (b -> e)
            (arrow(union(a, b), intersect(c, d, e)), intersect(
                arrow(a, c),
                arrow(a, d),
                arrow(a, e),
                arrow(b, c),
                arrow(b, d),
                arrow(b, e)
            )),

            # (a ∨ (b ∧ c)) -> (d ∧ (e ∨ f)) => (a -> d) ∧ (a -> (e ∨ f)) ∧ ((b ∧ c) -> d) ∧ ((b ∧ c) -> (e ∨ f))
            (arrow(union(a, intersect(b, c)), intersect(d, union(e, f))), intersect(
                arrow(a, d),
                arrow(a, union(e, f)),
                arrow(intersect(b, c), d),
                arrow(intersect(b, c), union(e, f))
            )),

            # (a ∧ (b ∨ c)) → (d ∨ (e ∧ f)) =>
            # ((a ∧ b) → (d ∨ e)) ∧ ((a ∧ b) → (d ∨ f)) ∧ ((a ∧ c) → (d ∨ e)) ∧ ((a ∧ c) → (d ∨ f))
            (arrow(intersect(a, union(b, c)), union(d, intersect(e, f))), intersect(
                arrow(intersect(a, b), union(d, e)),
                arrow(intersect(a, b), union(d, f)),
                arrow(intersect(a, c), union(d, e)),
                arrow(intersect(a, c), union(d, f)),
            )),
        ]
        # self.test_single_case(test_cases, 10)
        #
        for index, (test_case, expected_result) in enumerate(test_cases):
            print(f"Index: {index}")
            print(test_case)
            print(expected_result)
            converted = convert_to_anf(test_case)
            assert_equal(converted, expected_result)

    def test_single_case(self, test_cases, current_test_case):
        print("\nTesting case: " + str(current_test_case))
        print(test_cases[current_test_case][0])
        print(test_cases[current_test_case][1])
        converted = convert_to_anf(test_cases[current_test_case][0])
        print("\noriginal:")
        print(test_cases[current_test_case][0])
        print("converted:")
        print(converted)
        print("expected:")
        print(test_cases[current_test_case][1])
        assert_equal(converted, test_cases[current_test_case][1])

    def test_is_BCDd95_subtype(self) -> None:
        fx = self.fx_co

        def intersect(*a: Type) -> IntersectionType:
            return IntersectionType(list(a))

        def union(*a: Type) -> UnionType:
            return UnionType(list(a))

        def arrow(self, *a: Type) -> CallableType:
            return fx.callable(self, *a)

        # - σ → τ rewrites to DNF(σ) → CNF(τ);
        # – ∪iσi → ∩j τj rewrites to ∩i(∩j(σi → τj )).
        omega = fx.anyt
        a = fx.a
        b = fx.b
        c = fx.c
        d = fx.d
        e = fx.e
        f = fx.f

        a1 = fx.a1  # class A1 inherits from A2
        a2 = fx.a2
        b1 = fx.b1  # class B1 inherits from B2
        b2 = fx.b2

        d1 = fx.d1  # d1 inherits from e1
        e1 = fx.e1  # e1 inherits from f1
        f1 = fx.f1

        test_cases = [
            # Testing the BCDd95 subtyping axioms
            # a <= a ∧ a
            (a, intersect(a, a)),

            # a ∨ a <= a
            (union(a, a), a),

            # a ∧ b <= a
            (intersect(a, b), a),

            # a ∧ b <= b
            (intersect(a, b), b),

            # a <= a ∨ b
            (a, union(a, b)),

            # b <= a ∨ b
            (b, union(a, b)),

            # a <= omega
            (a, omega),

            # a <= a
            (a, a),

            # a1 <= a2, b1 <= b2 => a1 ∧ b1 <=  a2 ∧ b2
            (intersect(a1, b1), intersect(a2, b2)),

            # a1 <= a2, b1 <= b2 => a1 ∨ b1 <=  a2 ∨ b2
            (union(a1, b1), union(a2, b2)),

            # d1 <= e1, e1 <= f1 => d1 <= f1 TODO OMAR: this doesnt work with AnyType
            (d1, f1),

            # (a ∧ (b ∨ c)) <= ((a ∧ b) ∨ (a ∧ c))
            (intersect(a, union(b, c)), union(intersect(a, b), intersect(a, c))),

            # a -> b ∧ a -> c <= a -> (b ∧ c)
            (intersect(arrow(a, b), arrow(a, c)),  arrow(a, intersect(b, c))),

            # a -> c ∧ b -> c <=  (a ∨ b) -> c
            (intersect(arrow(a, c), arrow(b, c)), arrow(union(a, b), c)),

            # ω <= ω -> ω
            (omega, arrow(omega, omega)),

            # a1 <= a2, b1 <= b2 => a2 -> b1 <= a1 -> b2
            (arrow(a2, b1), arrow(a1, b2)),

            # (a ∧ (b ∨ c)) → (d ∨ (e ∧ f)) <= omega
            (arrow(intersect(a, union(b, c)), union(d, intersect(e, f))), omega),
            # (a ∧ (b ∨ c)) → (d ∨ (e ∧ f)) <=
            # ((a ∧ b) → (d ∨ e)) ∧ ((a ∧ b) → (d ∨ f)) ∧ ((a ∧ c) → (d ∨ e)) ∧ ((a ∧ c) → (d ∨ f))
            (a, intersect(a, a)),
            (arrow(intersect(a, union(b, c)), union(d, intersect(e, f))), intersect(
                arrow(intersect(a, b), union(d, e)),
                arrow(intersect(a, b), union(d, f)),
                arrow(intersect(a, c), union(d, e)),
                arrow(intersect(a, c), union(d, f)),
            )),

            # (a ∨ (b ∧ c)) -> (d ∧ (e ∨ f)) => (a -> d) ∧ (a -> (e ∨ f)) ∧ ((b ∧ c) -> d) ∧ ((b ∧ c) -> (e ∨ f))
            (arrow(union(a, intersect(b, c)), intersect(d, union(e, f))), intersect(
                arrow(a, d),
                arrow(a, union(e, f)),
                arrow(intersect(b, c), d),
                arrow(intersect(b, c), union(e, f))
            )),

            # (a -> d) ∧ (a -> (e ∨ f)) ∧ ((b ∧ c) -> d) ∧ ((b ∧ c) -> (e ∨ f)) => (a ∨ (b ∧ c)) -> (d ∧ (e ∨ f))
            (intersect(
                arrow(a, d),
                arrow(a, union(e, f)),
                arrow(intersect(b, c), d),
                arrow(intersect(b, c), union(e, f))
            ), arrow(union(a, intersect(b, c)), intersect(d, union(e, f)))),
        ]
        # current_test_case = 15
        # print("\nTesting case: " + str(current_test_case))
        # print(test_cases[current_test_case][0])
        # print(test_cases[current_test_case][1])
        # assert is_BCDd95_subtype(test_cases[current_test_case][0], test_cases[current_test_case][1])

        for index, (left, right) in enumerate(test_cases):
            print(f"Index: {index}")
            assert is_xi_subtype(left, right)


    def test_intersection(self) -> None:
        fx = self.fx_co

        lit1 = fx.lit1
        lit2 = fx.lit2
        lit3 = fx.lit3
        # top = fx.o
        top = fx.anyt

        # # # Test BCD83 Rules # # #

        # σ ≤ ω
        assert is_subtype(lit1, fx.o)

        # ω ≤ ω → ω
        assert is_subtype(top, fx.callable(top, top))

        # σ ∩ τ ≤ σ, σ ∩ τ ≤ τ
        self.assert_not_subtype(self.intersection_type(fx.d), fx.a)
        assert is_subtype(self.intersection_type(fx.a, fx.d), fx.a)
        assert is_subtype(self.intersection_type(fx.a, fx.d), fx.d)
        assert is_subtype(self.fx.ad, self.fx.a)
        assert is_subtype(self.fx.ad, self.fx.d)

        # (σ → τ1) ∩ (σ → τ2) ≤ σ → τ1 ∩ τ2
        assert is_subtype(
            self.intersection_type(
                fx.callable(fx.sigma_one, fx.tau_one),
                fx.callable(fx.sigma_one, fx.tau_two)),
            fx.callable(fx.sigma_one, self.intersection_type(fx.tau_one, fx.tau_two))
        )
        self.assert_not_arrow_cases(fx)

        # σ ≤ τ1 ∧ σ ≤ τ2 ⇒ σ ≤ τ1 ∩ τ2
        assert is_subtype(self.fx.ad, self.intersection_type(self.fx.a, self.fx.d))

        # σ2 ≤ σ1 ∧ τ1 ≤ τ2 ⇒ σ1 → τ1 ≤ σ2 → τ2
        # covariance and contravariance check
        sigma_one = self.fx.vehicle
        sigma_two = self.fx.car
        tau_one = self.fx.manager
        tau_two = self.fx.employee
        assert is_subtype(sigma_two, sigma_one) and is_subtype(tau_one, tau_two)
        assert is_subtype(
            fx.callable(sigma_one, tau_one),
            fx.callable(sigma_two, tau_two)
        )

        # TODO OMAR: add subtyping without inheritance through abc
        #   https://stackoverflow.com/questions/38275148/python-subclass-that-doesnt-inherit-attributes

    def assert_not_arrow_cases(self, fx):
        assert not is_subtype(
            self.intersection_type(
                fx.callable(fx.sigma_one, fx.tau_one),
                fx.callable(fx.sigma_one, fx.sigma_two)),
            fx.callable(fx.sigma_one, self.intersection_type(fx.tau_one, fx.tau_two))
        )
        assert not is_subtype(
            self.intersection_type(
                fx.callable(fx.sigma_one, fx.tau_one),
                fx.callable(fx.sigma_one, fx.tau_two)),
            fx.callable(fx.sigma_one, self.intersection_type(fx.sigma_two, fx.tau_two))
        )
        assert not is_subtype(
            self.intersection_type(
                fx.callable(fx.sigma_one, fx.tau_one),
                fx.callable(fx.sigma_one, fx.tau_two)),
            fx.callable(fx.sigma_two, self.intersection_type(fx.tau_one, fx.tau_two))
        )

    def test_is_proper_subtype_covariance(self) -> None:
        fx_co = self.fx_co

        assert is_proper_subtype(fx_co.gsab, fx_co.gb)
        assert is_proper_subtype(fx_co.gsab, fx_co.ga)
        assert not is_proper_subtype(fx_co.gsaa, fx_co.gb)
        assert is_proper_subtype(fx_co.gb, fx_co.ga)
        assert not is_proper_subtype(fx_co.ga, fx_co.gb)

    def test_is_proper_subtype_contravariance(self) -> None:
        fx_contra = self.fx_contra

        assert is_proper_subtype(fx_contra.gsab, fx_contra.gb)
        assert not is_proper_subtype(fx_contra.gsab, fx_contra.ga)
        assert is_proper_subtype(fx_contra.gsaa, fx_contra.gb)
        assert not is_proper_subtype(fx_contra.gb, fx_contra.ga)
        assert is_proper_subtype(fx_contra.ga, fx_contra.gb)

    def test_is_proper_subtype_invariance(self) -> None:
        fx = self.fx

        assert is_proper_subtype(fx.gsab, fx.gb)
        assert not is_proper_subtype(fx.gsab, fx.ga)
        assert not is_proper_subtype(fx.gsaa, fx.gb)
        assert not is_proper_subtype(fx.gb, fx.ga)
        assert not is_proper_subtype(fx.ga, fx.gb)

    def test_is_proper_subtype_and_subtype_literal_types(self) -> None:
        fx = self.fx

        lit1 = fx.lit1
        lit2 = fx.lit2
        lit3 = fx.lit3

        assert is_proper_subtype(lit1, fx.a)
        assert not is_proper_subtype(lit1, fx.d)
        assert not is_proper_subtype(fx.a, lit1)
        assert is_proper_subtype(fx.uninhabited, lit1)
        assert not is_proper_subtype(lit1, fx.uninhabited)
        assert is_proper_subtype(lit1, lit1)
        assert not is_proper_subtype(lit1, lit2)
        assert not is_proper_subtype(lit2, lit3)

        assert is_subtype(lit1, fx.a)
        assert not is_subtype(lit1, fx.d)
        assert not is_subtype(fx.a, lit1)
        assert is_subtype(fx.uninhabited, lit1)
        assert not is_subtype(lit1, fx.uninhabited)
        assert is_subtype(lit1, lit1)
        assert not is_subtype(lit1, lit2)
        assert not is_subtype(lit2, lit3)

        assert not is_proper_subtype(lit1, fx.anyt)
        assert not is_proper_subtype(fx.anyt, lit1)

        assert is_subtype(lit1, fx.anyt)
        assert is_subtype(fx.anyt, lit1)

    def test_subtype_aliases(self) -> None:
        A1, _ = self.fx.def_alias_1(self.fx.a)
        AA1, _ = self.fx.def_alias_1(self.fx.a)
        assert is_subtype(A1, AA1)
        assert is_subtype(AA1, A1)

        A2, _ = self.fx.def_alias_2(self.fx.a)
        AA2, _ = self.fx.def_alias_2(self.fx.a)
        assert is_subtype(A2, AA2)
        assert is_subtype(AA2, A2)

        B1, _ = self.fx.def_alias_1(self.fx.b)
        B2, _ = self.fx.def_alias_2(self.fx.b)
        assert is_subtype(B1, A1)
        assert is_subtype(B2, A2)
        assert not is_subtype(A1, B1)
        assert not is_subtype(A2, B2)

        assert not is_subtype(A2, A1)
        assert is_subtype(A1, A2)

    # can_be_true / can_be_false

    def test_empty_tuple_always_false(self) -> None:
        tuple_type = self.tuple()
        assert tuple_type.can_be_false
        assert not tuple_type.can_be_true

    def test_nonempty_tuple_always_true(self) -> None:
        tuple_type = self.tuple(AnyType(TypeOfAny.special_form), AnyType(TypeOfAny.special_form))
        assert tuple_type.can_be_true
        assert not tuple_type.can_be_false

    def test_union_can_be_true_if_any_true(self) -> None:
        union_type = UnionType([self.fx.a, self.tuple()])
        assert union_type.can_be_true

    def test_union_can_not_be_true_if_none_true(self) -> None:
        union_type = UnionType([self.tuple(), self.tuple()])
        assert not union_type.can_be_true

    def test_union_can_be_false_if_any_false(self) -> None:
        union_type = UnionType([self.fx.a, self.tuple()])
        assert union_type.can_be_false

    def test_union_can_not_be_false_if_none_false(self) -> None:
        union_type = UnionType([self.tuple(self.fx.a), self.tuple(self.fx.d)])
        assert not union_type.can_be_false

    # true_only / false_only

    def test_true_only_of_false_type_is_uninhabited(self) -> None:
        to = true_only(NoneType())
        assert_type(UninhabitedType, to)

    def test_true_only_of_true_type_is_idempotent(self) -> None:
        always_true = self.tuple(AnyType(TypeOfAny.special_form))
        to = true_only(always_true)
        assert always_true is to

    def test_true_only_of_instance(self) -> None:
        to = true_only(self.fx.a)
        assert_equal(str(to), "A")
        assert to.can_be_true
        assert not to.can_be_false
        assert_type(Instance, to)
        # The original class still can be false
        assert self.fx.a.can_be_false

    def test_true_only_of_union(self) -> None:
        tup_type = self.tuple(AnyType(TypeOfAny.special_form))
        # Union of something that is unknown, something that is always true, something
        # that is always false
        union_type = UnionType([self.fx.a, tup_type, self.tuple()])
        to = true_only(union_type)
        assert isinstance(to, UnionType)
        assert_equal(len(to.items), 2)
        assert to.items[0].can_be_true
        assert not to.items[0].can_be_false
        assert to.items[1] is tup_type

    def test_false_only_of_true_type_is_uninhabited(self) -> None:
        with state.strict_optional_set(True):
            fo = false_only(self.tuple(AnyType(TypeOfAny.special_form)))
            assert_type(UninhabitedType, fo)

    def test_false_only_tuple(self) -> None:
        with state.strict_optional_set(False):
            fo = false_only(self.tuple(self.fx.a))
            assert_equal(fo, NoneType())
        with state.strict_optional_set(True):
            fo = false_only(self.tuple(self.fx.a))
            assert_equal(fo, UninhabitedType())

    def test_false_only_of_false_type_is_idempotent(self) -> None:
        always_false = NoneType()
        fo = false_only(always_false)
        assert always_false is fo

    def test_false_only_of_instance(self) -> None:
        fo = false_only(self.fx.a)
        assert_equal(str(fo), "A")
        assert not fo.can_be_true
        assert fo.can_be_false
        assert_type(Instance, fo)
        # The original class still can be true
        assert self.fx.a.can_be_true

    def test_false_only_of_union(self) -> None:
        with state.strict_optional_set(True):
            tup_type = self.tuple()
            # Union of something that is unknown, something that is always true, something
            # that is always false
            union_type = UnionType(
                [self.fx.a, self.tuple(AnyType(TypeOfAny.special_form)), tup_type]
            )
            assert_equal(len(union_type.items), 3)
            fo = false_only(union_type)
            assert isinstance(fo, UnionType)
            assert_equal(len(fo.items), 2)
            assert not fo.items[0].can_be_true
            assert fo.items[0].can_be_false
            assert fo.items[1] is tup_type

    def test_simplified_union(self) -> None:
        fx = self.fx

        self.assert_simplified_union([fx.a, fx.a], fx.a)
        self.assert_simplified_union([fx.a, fx.b], fx.a)
        self.assert_simplified_union([fx.a, fx.d], UnionType([fx.a, fx.d]))
        self.assert_simplified_union([fx.a, fx.uninhabited], fx.a)
        self.assert_simplified_union([fx.ga, fx.gs2a], fx.ga)
        self.assert_simplified_union([fx.ga, fx.gsab], UnionType([fx.ga, fx.gsab]))
        self.assert_simplified_union([fx.ga, fx.gsba], fx.ga)
        self.assert_simplified_union([fx.a, UnionType([fx.d])], UnionType([fx.a, fx.d]))
        self.assert_simplified_union([fx.a, UnionType([fx.a])], fx.a)
        self.assert_simplified_union(
            [fx.b, UnionType([fx.c, UnionType([fx.d])])], UnionType([fx.b, fx.c, fx.d])
        )

    def test_simplified_union_with_literals(self) -> None:
        fx = self.fx

        self.assert_simplified_union([fx.lit1, fx.a], fx.a)
        self.assert_simplified_union([fx.lit1, fx.lit2, fx.a], fx.a)
        self.assert_simplified_union([fx.lit1, fx.lit1], fx.lit1)
        self.assert_simplified_union([fx.lit1, fx.lit2], UnionType([fx.lit1, fx.lit2]))
        self.assert_simplified_union([fx.lit1, fx.lit3], UnionType([fx.lit1, fx.lit3]))
        self.assert_simplified_union([fx.lit1, fx.uninhabited], fx.lit1)
        self.assert_simplified_union([fx.lit1_inst, fx.a], fx.a)
        self.assert_simplified_union([fx.lit1_inst, fx.lit1_inst], fx.lit1_inst)
        self.assert_simplified_union(
            [fx.lit1_inst, fx.lit2_inst], UnionType([fx.lit1_inst, fx.lit2_inst])
        )
        self.assert_simplified_union(
            [fx.lit1_inst, fx.lit3_inst], UnionType([fx.lit1_inst, fx.lit3_inst])
        )
        self.assert_simplified_union([fx.lit1_inst, fx.uninhabited], fx.lit1_inst)
        self.assert_simplified_union([fx.lit1, fx.lit1_inst], fx.lit1)
        self.assert_simplified_union([fx.lit1, fx.lit2_inst], UnionType([fx.lit1, fx.lit2_inst]))
        self.assert_simplified_union([fx.lit1, fx.lit3_inst], UnionType([fx.lit1, fx.lit3_inst]))

    def test_simplified_union_with_str_literals(self) -> None:
        fx = self.fx

        self.assert_simplified_union([fx.lit_str1, fx.lit_str2, fx.str_type], fx.str_type)
        self.assert_simplified_union([fx.lit_str1, fx.lit_str1, fx.lit_str1], fx.lit_str1)
        self.assert_simplified_union(
            [fx.lit_str1, fx.lit_str2, fx.lit_str3],
            UnionType([fx.lit_str1, fx.lit_str2, fx.lit_str3]),
        )
        self.assert_simplified_union(
            [fx.lit_str1, fx.lit_str2, fx.uninhabited], UnionType([fx.lit_str1, fx.lit_str2])
        )

    def test_simplify_very_large_union(self) -> None:
        fx = self.fx
        literals = []
        for i in range(5000):
            literals.append(LiteralType("v%d" % i, fx.str_type))
        # This shouldn't be very slow, even if the union is big.
        self.assert_simplified_union([*literals, fx.str_type], fx.str_type)

    def test_simplified_union_with_str_instance_literals(self) -> None:
        fx = self.fx

        self.assert_simplified_union(
            [fx.lit_str1_inst, fx.lit_str2_inst, fx.str_type], fx.str_type
        )
        self.assert_simplified_union(
            [fx.lit_str1_inst, fx.lit_str1_inst, fx.lit_str1_inst], fx.lit_str1_inst
        )
        self.assert_simplified_union(
            [fx.lit_str1_inst, fx.lit_str2_inst, fx.lit_str3_inst],
            UnionType([fx.lit_str1_inst, fx.lit_str2_inst, fx.lit_str3_inst]),
        )
        self.assert_simplified_union(
            [fx.lit_str1_inst, fx.lit_str2_inst, fx.uninhabited],
            UnionType([fx.lit_str1_inst, fx.lit_str2_inst]),
        )

    def test_simplified_union_with_mixed_str_literals(self) -> None:
        fx = self.fx

        self.assert_simplified_union(
            [fx.lit_str1, fx.lit_str2, fx.lit_str3_inst],
            UnionType([fx.lit_str1, fx.lit_str2, fx.lit_str3_inst]),
        )
        self.assert_simplified_union(
            [fx.lit_str1, fx.lit_str1, fx.lit_str1_inst],
            UnionType([fx.lit_str1, fx.lit_str1_inst]),
        )

    def assert_simplified_union(self, original: list[Type], union: Type) -> None:
        assert_equal(make_simplified_union(original), union)
        assert_equal(make_simplified_union(list(reversed(original))), union)

    # Helpers

    def tuple(self, *a: Type) -> TupleType:
        return TupleType(list(a), self.fx.std_tuple)

    def intersection_type(self, *a: Type) -> IntersectionType:
        return IntersectionType(list(a))

    def assert_subtype(self, s: Type, t: Type) -> None:
        assert is_subtype(s, t), f"{s} not subtype of {t}"

    def assert_not_subtype(self, s: Type, t: Type) -> None:
        assert not is_subtype(s, t), f"{s} subtype of {t}"

    def callable(self, vars: list[str], *a: Type) -> CallableType:
        """callable(args, a1, ..., an, r) constructs a callable with
        argument types a1, ... an and return type r and type arguments
        vars.
        """
        tv: list[TypeVarType] = []
        n = -1
        for v in vars:
            tv.append(TypeVarType(v, v, n, [], self.fx.o))
            n -= 1
        return CallableType(
            list(a[:-1]),
            [ARG_POS] * (len(a) - 1),
            [None] * (len(a) - 1),
            a[-1],
            self.fx.function,
            name=None,
            variables=tv,
        )


class JoinSuite(Suite):
    def setUp(self) -> None:
        self.fx = TypeFixture(INVARIANT)
        self.fx_co = TypeFixture(COVARIANT)
        self.fx_contra = TypeFixture(CONTRAVARIANT)

    def test_trivial_cases(self) -> None:
        for simple in self.fx.a, self.fx.o, self.fx.b:
            self.assert_join(simple, simple, simple)

    def test_class_subtyping(self) -> None:
        self.assert_join(self.fx.a, self.fx.o, self.fx.o)
        self.assert_join(self.fx.b, self.fx.o, self.fx.o)
        self.assert_join(self.fx.a, self.fx.d, self.fx.o)
        self.assert_join(self.fx.b, self.fx.c, self.fx.a)
        self.assert_join(self.fx.b, self.fx.d, self.fx.o)

    def test_tuples(self) -> None:
        self.assert_join(self.tuple(), self.tuple(), self.tuple())
        self.assert_join(self.tuple(self.fx.a), self.tuple(self.fx.a), self.tuple(self.fx.a))
        self.assert_join(
            self.tuple(self.fx.b, self.fx.c),
            self.tuple(self.fx.a, self.fx.d),
            self.tuple(self.fx.a, self.fx.o),
        )

        self.assert_join(
            self.tuple(self.fx.a, self.fx.a), self.fx.std_tuple, self.var_tuple(self.fx.anyt)
        )
        self.assert_join(
            self.tuple(self.fx.a), self.tuple(self.fx.a, self.fx.a), self.var_tuple(self.fx.a)
        )
        self.assert_join(
            self.tuple(self.fx.b), self.tuple(self.fx.a, self.fx.c), self.var_tuple(self.fx.a)
        )
        self.assert_join(self.tuple(), self.tuple(self.fx.a), self.var_tuple(self.fx.a))

    def test_var_tuples(self) -> None:
        self.assert_join(
            self.tuple(self.fx.a), self.var_tuple(self.fx.a), self.var_tuple(self.fx.a)
        )
        self.assert_join(
            self.var_tuple(self.fx.a), self.tuple(self.fx.a), self.var_tuple(self.fx.a)
        )
        self.assert_join(self.var_tuple(self.fx.a), self.tuple(), self.var_tuple(self.fx.a))

    def test_function_types(self) -> None:
        self.assert_join(
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.a, self.fx.b),
        )

        self.assert_join(
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.b, self.fx.b),
            self.callable(self.fx.b, self.fx.b),
        )
        self.assert_join(
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.a, self.fx.a),
            self.callable(self.fx.a, self.fx.a),
        )
        self.assert_join(self.callable(self.fx.a, self.fx.b), self.fx.function, self.fx.function)
        self.assert_join(
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.d, self.fx.b),
            self.fx.function,
        )

    def test_type_vars(self) -> None:
        self.assert_join(self.fx.t, self.fx.t, self.fx.t)
        self.assert_join(self.fx.s, self.fx.s, self.fx.s)
        self.assert_join(self.fx.t, self.fx.s, self.fx.o)

    def test_none(self) -> None:
        # Any type t joined with None results in t.
        for t in [
            NoneType(),
            self.fx.a,
            self.fx.o,
            UnboundType("x"),
            self.fx.t,
            self.tuple(),
            self.callable(self.fx.a, self.fx.b),
            self.fx.anyt,
        ]:
            self.assert_join(t, NoneType(), t)

    def test_unbound_type(self) -> None:
        self.assert_join(UnboundType("x"), UnboundType("x"), self.fx.anyt)
        self.assert_join(UnboundType("x"), UnboundType("y"), self.fx.anyt)

        # Any type t joined with an unbound type results in dynamic. Unbound
        # type means that there is an error somewhere in the program, so this
        # does not affect type safety (whatever the result).
        for t in [
            self.fx.a,
            self.fx.o,
            self.fx.ga,
            self.fx.t,
            self.tuple(),
            self.callable(self.fx.a, self.fx.b),
        ]:
            self.assert_join(t, UnboundType("X"), self.fx.anyt)

    def test_any_type(self) -> None:
        # Join against 'Any' type always results in 'Any'.
        for t in [
            self.fx.anyt,
            self.fx.a,
            self.fx.o,
            NoneType(),
            UnboundType("x"),
            self.fx.t,
            self.tuple(),
            self.callable(self.fx.a, self.fx.b),
        ]:
            self.assert_join(t, self.fx.anyt, self.fx.anyt)

    def test_mixed_truth_restricted_type_simple(self) -> None:
        # join_simple against differently restricted truthiness types drops restrictions.
        true_a = true_only(self.fx.a)
        false_o = false_only(self.fx.o)
        j = join_simple(self.fx.o, true_a, false_o)
        assert j.can_be_true
        assert j.can_be_false

    def test_mixed_truth_restricted_type(self) -> None:
        # join_types against differently restricted truthiness types drops restrictions.
        true_any = true_only(AnyType(TypeOfAny.special_form))
        false_o = false_only(self.fx.o)
        j = join_types(true_any, false_o)
        assert j.can_be_true
        assert j.can_be_false

    def test_other_mixed_types(self) -> None:
        # In general, joining unrelated types produces object.
        for t1 in [self.fx.a, self.fx.t, self.tuple(), self.callable(self.fx.a, self.fx.b)]:
            for t2 in [self.fx.a, self.fx.t, self.tuple(), self.callable(self.fx.a, self.fx.b)]:
                if str(t1) != str(t2):
                    self.assert_join(t1, t2, self.fx.o)

    def test_simple_generics(self) -> None:
        self.assert_join(self.fx.ga, self.fx.nonet, self.fx.ga)
        self.assert_join(self.fx.ga, self.fx.anyt, self.fx.anyt)

        for t in [
            self.fx.a,
            self.fx.o,
            self.fx.t,
            self.tuple(),
            self.callable(self.fx.a, self.fx.b),
        ]:
            self.assert_join(t, self.fx.ga, self.fx.o)

    def test_generics_invariant(self) -> None:
        self.assert_join(self.fx.ga, self.fx.ga, self.fx.ga)
        self.assert_join(self.fx.ga, self.fx.gb, self.fx.o)
        self.assert_join(self.fx.ga, self.fx.gd, self.fx.o)
        self.assert_join(self.fx.ga, self.fx.g2a, self.fx.o)

    def test_generics_covariant(self) -> None:
        self.assert_join(self.fx_co.ga, self.fx_co.ga, self.fx_co.ga)
        self.assert_join(self.fx_co.ga, self.fx_co.gb, self.fx_co.ga)
        self.assert_join(self.fx_co.ga, self.fx_co.gd, self.fx_co.go)
        self.assert_join(self.fx_co.ga, self.fx_co.g2a, self.fx_co.o)

    def test_generics_contravariant(self) -> None:
        self.assert_join(self.fx_contra.ga, self.fx_contra.ga, self.fx_contra.ga)
        # TODO: this can be more precise than "object", see a comment in mypy/join.py
        self.assert_join(self.fx_contra.ga, self.fx_contra.gb, self.fx_contra.o)
        self.assert_join(self.fx_contra.ga, self.fx_contra.g2a, self.fx_contra.o)

    def test_generics_with_multiple_args(self) -> None:
        self.assert_join(self.fx_co.hab, self.fx_co.hab, self.fx_co.hab)
        self.assert_join(self.fx_co.hab, self.fx_co.hbb, self.fx_co.hab)
        self.assert_join(self.fx_co.had, self.fx_co.haa, self.fx_co.hao)

    def test_generics_with_inheritance(self) -> None:
        self.assert_join(self.fx_co.gsab, self.fx_co.gb, self.fx_co.gb)
        self.assert_join(self.fx_co.gsba, self.fx_co.gb, self.fx_co.ga)
        self.assert_join(self.fx_co.gsab, self.fx_co.gd, self.fx_co.go)

    def test_generics_with_inheritance_and_shared_supertype(self) -> None:
        self.assert_join(self.fx_co.gsba, self.fx_co.gs2a, self.fx_co.ga)
        self.assert_join(self.fx_co.gsab, self.fx_co.gs2a, self.fx_co.ga)
        self.assert_join(self.fx_co.gsab, self.fx_co.gs2d, self.fx_co.go)

    def test_generic_types_and_any(self) -> None:
        self.assert_join(self.fx.gdyn, self.fx.ga, self.fx.gdyn)
        self.assert_join(self.fx_co.gdyn, self.fx_co.ga, self.fx_co.gdyn)
        self.assert_join(self.fx_contra.gdyn, self.fx_contra.ga, self.fx_contra.gdyn)

    def test_callables_with_any(self) -> None:
        self.assert_join(
            self.callable(self.fx.a, self.fx.a, self.fx.anyt, self.fx.a),
            self.callable(self.fx.a, self.fx.anyt, self.fx.a, self.fx.anyt),
            self.callable(self.fx.a, self.fx.anyt, self.fx.anyt, self.fx.anyt),
        )

    def test_overloaded(self) -> None:
        c = self.callable

        def ov(*items: CallableType) -> Overloaded:
            return Overloaded(list(items))

        fx = self.fx
        func = fx.function
        c1 = c(fx.a, fx.a)
        c2 = c(fx.b, fx.b)
        c3 = c(fx.c, fx.c)
        self.assert_join(ov(c1, c2), c1, c1)
        self.assert_join(ov(c1, c2), c2, c2)
        self.assert_join(ov(c1, c2), ov(c1, c2), ov(c1, c2))
        self.assert_join(ov(c1, c2), ov(c1, c3), c1)
        self.assert_join(ov(c2, c1), ov(c3, c1), c1)
        self.assert_join(ov(c1, c2), c3, func)

    def test_overloaded_with_any(self) -> None:
        c = self.callable

        def ov(*items: CallableType) -> Overloaded:
            return Overloaded(list(items))

        fx = self.fx
        any = fx.anyt
        self.assert_join(ov(c(fx.a, fx.a), c(fx.b, fx.b)), c(any, fx.b), c(any, fx.b))
        self.assert_join(ov(c(fx.a, fx.a), c(any, fx.b)), c(fx.b, fx.b), c(any, fx.b))

    @skip
    def test_join_interface_types(self) -> None:
        self.assert_join(self.fx.f, self.fx.f, self.fx.f)
        self.assert_join(self.fx.f, self.fx.f2, self.fx.o)
        self.assert_join(self.fx.f, self.fx.f3, self.fx.f)

    @skip
    def test_join_interface_and_class_types(self) -> None:
        self.assert_join(self.fx.o, self.fx.f, self.fx.o)
        self.assert_join(self.fx.a, self.fx.f, self.fx.o)

        self.assert_join(self.fx.e, self.fx.f, self.fx.f)

    @skip
    def test_join_class_types_with_interface_result(self) -> None:
        # Unique result
        self.assert_join(self.fx.e, self.fx.e2, self.fx.f)

        # Ambiguous result
        self.assert_join(self.fx.e2, self.fx.e3, self.fx.anyt)

    @skip
    def test_generic_interfaces(self) -> None:
        fx = InterfaceTypeFixture()

        self.assert_join(fx.gfa, fx.gfa, fx.gfa)
        self.assert_join(fx.gfa, fx.gfb, fx.o)

        self.assert_join(fx.m1, fx.gfa, fx.gfa)

        self.assert_join(fx.m1, fx.gfb, fx.o)

    def test_simple_type_objects(self) -> None:
        t1 = self.type_callable(self.fx.a, self.fx.a)
        t2 = self.type_callable(self.fx.b, self.fx.b)
        tr = self.type_callable(self.fx.b, self.fx.a)

        self.assert_join(t1, t1, t1)
        j = join_types(t1, t1)
        assert isinstance(j, CallableType)
        assert j.is_type_obj()

        self.assert_join(t1, t2, tr)
        self.assert_join(t1, self.fx.type_type, self.fx.type_type)
        self.assert_join(self.fx.type_type, self.fx.type_type, self.fx.type_type)

    def test_type_type(self) -> None:
        self.assert_join(self.fx.type_a, self.fx.type_b, self.fx.type_a)
        self.assert_join(self.fx.type_b, self.fx.type_any, self.fx.type_any)
        self.assert_join(self.fx.type_b, self.fx.type_type, self.fx.type_type)
        self.assert_join(self.fx.type_b, self.fx.type_c, self.fx.type_a)
        self.assert_join(self.fx.type_c, self.fx.type_d, TypeType.make_normalized(self.fx.o))
        self.assert_join(self.fx.type_type, self.fx.type_any, self.fx.type_type)
        self.assert_join(self.fx.type_b, self.fx.anyt, self.fx.anyt)

    def test_literal_type(self) -> None:
        a = self.fx.a
        d = self.fx.d
        lit1 = self.fx.lit1
        lit2 = self.fx.lit2
        lit3 = self.fx.lit3

        self.assert_join(lit1, lit1, lit1)
        self.assert_join(lit1, a, a)
        self.assert_join(lit1, d, self.fx.o)
        self.assert_join(lit1, lit2, a)
        self.assert_join(lit1, lit3, self.fx.o)
        self.assert_join(lit1, self.fx.anyt, self.fx.anyt)
        self.assert_join(UnionType([lit1, lit2]), lit2, UnionType([lit1, lit2]))
        self.assert_join(UnionType([lit1, lit2]), a, a)
        self.assert_join(UnionType([lit1, lit3]), a, UnionType([a, lit3]))
        self.assert_join(UnionType([d, lit3]), lit3, d)
        self.assert_join(UnionType([d, lit3]), d, UnionType([d, lit3]))
        self.assert_join(UnionType([a, lit1]), lit1, a)
        self.assert_join(UnionType([a, lit1]), lit2, a)
        self.assert_join(UnionType([lit1, lit2]), UnionType([lit1, lit2]), UnionType([lit1, lit2]))

        # The order in which we try joining two unions influences the
        # ordering of the items in the final produced unions. So, we
        # manually call 'assert_simple_join' and tune the output
        # after swapping the arguments here.
        self.assert_simple_join(
            UnionType([lit1, lit2]), UnionType([lit2, lit3]), UnionType([lit1, lit2, lit3])
        )
        self.assert_simple_join(
            UnionType([lit2, lit3]), UnionType([lit1, lit2]), UnionType([lit2, lit3, lit1])
        )

    # There are additional test cases in check-inference.test.

    # TODO: Function types + varargs and default args.

    def assert_join(self, s: Type, t: Type, join: Type) -> None:
        self.assert_simple_join(s, t, join)
        self.assert_simple_join(t, s, join)

    def assert_simple_join(self, s: Type, t: Type, join: Type) -> None:
        result = join_types(s, t)
        actual = str(result)
        expected = str(join)
        assert_equal(actual, expected, f"join({s}, {t}) == {{}} ({{}} expected)")
        assert is_subtype(s, result), f"{s} not subtype of {result}"
        assert is_subtype(t, result), f"{t} not subtype of {result}"

    def tuple(self, *a: Type) -> TupleType:
        return TupleType(list(a), self.fx.std_tuple)

    def var_tuple(self, t: Type) -> Instance:
        """Construct a variable-length tuple type"""
        return Instance(self.fx.std_tuplei, [t])

    def callable(self, *a: Type) -> CallableType:
        """callable(a1, ..., an, r) constructs a callable with argument types
        a1, ... an and return type r.
        """
        n = len(a) - 1
        return CallableType(list(a[:-1]), [ARG_POS] * n, [None] * n, a[-1], self.fx.function)

    def type_callable(self, *a: Type) -> CallableType:
        """type_callable(a1, ..., an, r) constructs a callable with
        argument types a1, ... an and return type r, and which
        represents a type.
        """
        n = len(a) - 1
        return CallableType(list(a[:-1]), [ARG_POS] * n, [None] * n, a[-1], self.fx.type_type)


class MeetSuite(Suite):
    def setUp(self) -> None:
        self.fx = TypeFixture()

    def test_trivial_cases(self) -> None:
        for simple in self.fx.a, self.fx.o, self.fx.b:
            self.assert_meet(simple, simple, simple)

    def test_class_subtyping(self) -> None:
        self.assert_meet(self.fx.a, self.fx.o, self.fx.a)
        self.assert_meet(self.fx.a, self.fx.b, self.fx.b)
        self.assert_meet(self.fx.b, self.fx.o, self.fx.b)
        self.assert_meet(self.fx.a, self.fx.d, NoneType())
        self.assert_meet(self.fx.b, self.fx.c, NoneType())

    def test_tuples(self) -> None:
        self.assert_meet(self.tuple(), self.tuple(), self.tuple())
        self.assert_meet(self.tuple(self.fx.a), self.tuple(self.fx.a), self.tuple(self.fx.a))
        self.assert_meet(
            self.tuple(self.fx.b, self.fx.c),
            self.tuple(self.fx.a, self.fx.d),
            self.tuple(self.fx.b, NoneType()),
        )

        self.assert_meet(
            self.tuple(self.fx.a, self.fx.a), self.fx.std_tuple, self.tuple(self.fx.a, self.fx.a)
        )
        self.assert_meet(self.tuple(self.fx.a), self.tuple(self.fx.a, self.fx.a), NoneType())

    def test_function_types(self) -> None:
        self.assert_meet(
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.a, self.fx.b),
        )

        self.assert_meet(
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.b, self.fx.b),
            self.callable(self.fx.a, self.fx.b),
        )
        self.assert_meet(
            self.callable(self.fx.a, self.fx.b),
            self.callable(self.fx.a, self.fx.a),
            self.callable(self.fx.a, self.fx.b),
        )

    def test_type_vars(self) -> None:
        self.assert_meet(self.fx.t, self.fx.t, self.fx.t)
        self.assert_meet(self.fx.s, self.fx.s, self.fx.s)
        self.assert_meet(self.fx.t, self.fx.s, NoneType())

    def test_none(self) -> None:
        self.assert_meet(NoneType(), NoneType(), NoneType())

        self.assert_meet(NoneType(), self.fx.anyt, NoneType())

        # Any type t joined with None results in None, unless t is Any.
        for t in [
            self.fx.a,
            self.fx.o,
            UnboundType("x"),
            self.fx.t,
            self.tuple(),
            self.callable(self.fx.a, self.fx.b),
        ]:
            self.assert_meet(t, NoneType(), NoneType())

    def test_unbound_type(self) -> None:
        self.assert_meet(UnboundType("x"), UnboundType("x"), self.fx.anyt)
        self.assert_meet(UnboundType("x"), UnboundType("y"), self.fx.anyt)

        self.assert_meet(UnboundType("x"), self.fx.anyt, UnboundType("x"))

        # The meet of any type t with an unbound type results in dynamic.
        # Unbound type means that there is an error somewhere in the program,
        # so this does not affect type safety.
        for t in [
            self.fx.a,
            self.fx.o,
            self.fx.t,
            self.tuple(),
            self.callable(self.fx.a, self.fx.b),
        ]:
            self.assert_meet(t, UnboundType("X"), self.fx.anyt)

    def test_dynamic_type(self) -> None:
        # Meet against dynamic type always results in dynamic.
        for t in [
            self.fx.anyt,
            self.fx.a,
            self.fx.o,
            NoneType(),
            UnboundType("x"),
            self.fx.t,
            self.tuple(),
            self.callable(self.fx.a, self.fx.b),
        ]:
            self.assert_meet(t, self.fx.anyt, t)

    def test_simple_generics(self) -> None:
        self.assert_meet(self.fx.ga, self.fx.ga, self.fx.ga)
        self.assert_meet(self.fx.ga, self.fx.o, self.fx.ga)
        self.assert_meet(self.fx.ga, self.fx.gb, self.fx.gb)
        self.assert_meet(self.fx.ga, self.fx.gd, self.fx.nonet)
        self.assert_meet(self.fx.ga, self.fx.g2a, self.fx.nonet)

        self.assert_meet(self.fx.ga, self.fx.nonet, self.fx.nonet)
        self.assert_meet(self.fx.ga, self.fx.anyt, self.fx.ga)

        for t in [self.fx.a, self.fx.t, self.tuple(), self.callable(self.fx.a, self.fx.b)]:
            self.assert_meet(t, self.fx.ga, self.fx.nonet)

    def test_generics_with_multiple_args(self) -> None:
        self.assert_meet(self.fx.hab, self.fx.hab, self.fx.hab)
        self.assert_meet(self.fx.hab, self.fx.haa, self.fx.hab)
        self.assert_meet(self.fx.hab, self.fx.had, self.fx.nonet)
        self.assert_meet(self.fx.hab, self.fx.hbb, self.fx.hbb)

    def test_generics_with_inheritance(self) -> None:
        self.assert_meet(self.fx.gsab, self.fx.gb, self.fx.gsab)
        self.assert_meet(self.fx.gsba, self.fx.gb, self.fx.nonet)

    def test_generics_with_inheritance_and_shared_supertype(self) -> None:
        self.assert_meet(self.fx.gsba, self.fx.gs2a, self.fx.nonet)
        self.assert_meet(self.fx.gsab, self.fx.gs2a, self.fx.nonet)

    def test_generic_types_and_dynamic(self) -> None:
        self.assert_meet(self.fx.gdyn, self.fx.ga, self.fx.ga)

    def test_callables_with_dynamic(self) -> None:
        self.assert_meet(
            self.callable(self.fx.a, self.fx.a, self.fx.anyt, self.fx.a),
            self.callable(self.fx.a, self.fx.anyt, self.fx.a, self.fx.anyt),
            self.callable(self.fx.a, self.fx.anyt, self.fx.anyt, self.fx.anyt),
        )

    def test_meet_interface_types(self) -> None:
        self.assert_meet(self.fx.f, self.fx.f, self.fx.f)
        self.assert_meet(self.fx.f, self.fx.f2, self.fx.nonet)
        self.assert_meet(self.fx.f, self.fx.f3, self.fx.f3)

    def test_meet_interface_and_class_types(self) -> None:
        self.assert_meet(self.fx.o, self.fx.f, self.fx.f)
        self.assert_meet(self.fx.a, self.fx.f, self.fx.nonet)

        self.assert_meet(self.fx.e, self.fx.f, self.fx.e)

    def test_meet_class_types_with_shared_interfaces(self) -> None:
        # These have nothing special with respect to meets, unlike joins. These
        # are for completeness only.
        self.assert_meet(self.fx.e, self.fx.e2, self.fx.nonet)
        self.assert_meet(self.fx.e2, self.fx.e3, self.fx.nonet)

    @skip
    def test_meet_with_generic_interfaces(self) -> None:
        fx = InterfaceTypeFixture()
        self.assert_meet(fx.gfa, fx.m1, fx.m1)
        self.assert_meet(fx.gfa, fx.gfa, fx.gfa)
        self.assert_meet(fx.gfb, fx.m1, fx.nonet)

    def test_type_type(self) -> None:
        self.assert_meet(self.fx.type_a, self.fx.type_b, self.fx.type_b)
        self.assert_meet(self.fx.type_b, self.fx.type_any, self.fx.type_b)
        self.assert_meet(self.fx.type_b, self.fx.type_type, self.fx.type_b)
        self.assert_meet(self.fx.type_b, self.fx.type_c, self.fx.nonet)
        self.assert_meet(self.fx.type_c, self.fx.type_d, self.fx.nonet)
        self.assert_meet(self.fx.type_type, self.fx.type_any, self.fx.type_any)
        self.assert_meet(self.fx.type_b, self.fx.anyt, self.fx.type_b)

    def test_literal_type(self) -> None:
        a = self.fx.a
        lit1 = self.fx.lit1
        lit2 = self.fx.lit2
        lit3 = self.fx.lit3

        self.assert_meet(lit1, lit1, lit1)
        self.assert_meet(lit1, a, lit1)
        self.assert_meet_uninhabited(lit1, lit3)
        self.assert_meet_uninhabited(lit1, lit2)
        self.assert_meet(UnionType([lit1, lit2]), lit1, lit1)
        self.assert_meet(UnionType([lit1, lit2]), UnionType([lit2, lit3]), lit2)
        self.assert_meet(UnionType([lit1, lit2]), UnionType([lit1, lit2]), UnionType([lit1, lit2]))
        self.assert_meet(lit1, self.fx.anyt, lit1)
        self.assert_meet(lit1, self.fx.o, lit1)

        assert is_same_type(lit1, narrow_declared_type(lit1, a))
        assert is_same_type(lit2, narrow_declared_type(lit2, a))

    # FIX generic interfaces + ranges

    def assert_meet_uninhabited(self, s: Type, t: Type) -> None:
        with state.strict_optional_set(False):
            self.assert_meet(s, t, self.fx.nonet)
        with state.strict_optional_set(True):
            self.assert_meet(s, t, self.fx.uninhabited)

    def assert_meet(self, s: Type, t: Type, meet: Type) -> None:
        self.assert_simple_meet(s, t, meet)
        self.assert_simple_meet(t, s, meet)

    def assert_simple_meet(self, s: Type, t: Type, meet: Type) -> None:
        result = meet_types(s, t)
        actual = str(result)
        expected = str(meet)
        assert_equal(actual, expected, f"meet({s}, {t}) == {{}} ({{}} expected)")
        assert is_subtype(result, s), f"{result} not subtype of {s}"
        assert is_subtype(result, t), f"{result} not subtype of {t}"

    def tuple(self, *a: Type) -> TupleType:
        return TupleType(list(a), self.fx.std_tuple)

    def callable(self, *a: Type) -> CallableType:
        """callable(a1, ..., an, r) constructs a callable with argument types
        a1, ... an and return type r.
        """
        n = len(a) - 1
        return CallableType(list(a[:-1]), [ARG_POS] * n, [None] * n, a[-1], self.fx.function)


class SameTypeSuite(Suite):
    def setUp(self) -> None:
        self.fx = TypeFixture()

    def test_literal_type(self) -> None:
        a = self.fx.a
        b = self.fx.b  # Reminder: b is a subclass of a

        lit1 = self.fx.lit1
        lit2 = self.fx.lit2
        lit3 = self.fx.lit3

        self.assert_same(lit1, lit1)
        self.assert_same(UnionType([lit1, lit2]), UnionType([lit1, lit2]))
        self.assert_same(UnionType([lit1, lit2]), UnionType([lit2, lit1]))
        self.assert_same(UnionType([a, b]), UnionType([b, a]))
        self.assert_not_same(lit1, b)
        self.assert_not_same(lit1, lit2)
        self.assert_not_same(lit1, lit3)

        self.assert_not_same(lit1, self.fx.anyt)
        self.assert_not_same(lit1, self.fx.nonet)

    def assert_same(self, s: Type, t: Type, strict: bool = True) -> None:
        self.assert_simple_is_same(s, t, expected=True, strict=strict)
        self.assert_simple_is_same(t, s, expected=True, strict=strict)

    def assert_not_same(self, s: Type, t: Type, strict: bool = True) -> None:
        self.assert_simple_is_same(s, t, False, strict=strict)
        self.assert_simple_is_same(t, s, False, strict=strict)

    def assert_simple_is_same(self, s: Type, t: Type, expected: bool, strict: bool) -> None:
        actual = is_same_type(s, t)
        assert_equal(actual, expected, f"is_same_type({s}, {t}) is {{}} ({{}} expected)")

        if strict:
            actual2 = s == t
            assert_equal(actual2, expected, f"({s} == {t}) is {{}} ({{}} expected)")
            assert_equal(
                hash(s) == hash(t), expected, f"(hash({s}) == hash({t}) is {{}} ({{}} expected)"
            )


class RemoveLastKnownValueSuite(Suite):
    def setUp(self) -> None:
        self.fx = TypeFixture()

    def test_optional(self) -> None:
        t = UnionType.make_union([self.fx.a, self.fx.nonet])
        self.assert_union_result(t, [self.fx.a, self.fx.nonet])

    def test_two_instances(self) -> None:
        t = UnionType.make_union([self.fx.a, self.fx.b])
        self.assert_union_result(t, [self.fx.a, self.fx.b])

    def test_multiple_same_instances(self) -> None:
        t = UnionType.make_union([self.fx.a, self.fx.a])
        assert remove_instance_last_known_values(t) == self.fx.a
        t = UnionType.make_union([self.fx.a, self.fx.a, self.fx.b])
        self.assert_union_result(t, [self.fx.a, self.fx.b])
        t = UnionType.make_union([self.fx.a, self.fx.nonet, self.fx.a, self.fx.b])
        self.assert_union_result(t, [self.fx.a, self.fx.nonet, self.fx.b])

    def test_single_last_known_value(self) -> None:
        t = UnionType.make_union([self.fx.lit1_inst, self.fx.nonet])
        self.assert_union_result(t, [self.fx.a, self.fx.nonet])

    def test_last_known_values_with_merge(self) -> None:
        t = UnionType.make_union([self.fx.lit1_inst, self.fx.lit2_inst, self.fx.lit4_inst])
        assert remove_instance_last_known_values(t) == self.fx.a
        t = UnionType.make_union(
            [self.fx.lit1_inst, self.fx.b, self.fx.lit2_inst, self.fx.lit4_inst]
        )
        self.assert_union_result(t, [self.fx.a, self.fx.b])

    def test_generics(self) -> None:
        t = UnionType.make_union([self.fx.ga, self.fx.gb])
        self.assert_union_result(t, [self.fx.ga, self.fx.gb])

    def assert_union_result(self, t: ProperType, expected: list[Type]) -> None:
        t2 = remove_instance_last_known_values(t)
        assert type(t2) is UnionType
        assert t2.items == expected

