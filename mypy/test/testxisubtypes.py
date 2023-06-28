"""Test cases for mypy types and type operations."""

from __future__ import annotations

from mypy.erasetype import erase_type, remove_instance_last_known_values
from mypy.nodes import ARG_OPT, ARG_POS, ARG_STAR, ARG_STAR2, CONTRAVARIANT, COVARIANT, INVARIANT
from mypy.subtypes import is_more_precise, is_proper_subtype, is_same_type, is_subtype, simplify_omega, \
    convert_to_cnf, convert_to_dnf, convert_to_anf, is_xi_subtype
from mypy.test.helpers import Suite, assert_equal, assert_type, skip
from mypy.test.typefixture import InterfaceTypeFixture, TypeFixture
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




class XiSubtypingSuite(Suite):
    def setUp(self) -> None:
        self.fx = TypeFixture(INVARIANT)
        self.fx_co = TypeFixture(COVARIANT)
        self.fx_contra = TypeFixture(CONTRAVARIANT)
    def test_simplify_omega(self) -> None:
        # ω ∩ σ and σ ∩ ω rewrite to σ;
        # ω ∪ σ and σ ∪ ω rewrite to ω;
        # σ → ω rewrites to ω.
        fx = self.fx_co
        omega = fx.anyt

        def intersect(*a: Type) -> IntersectionType:
            return IntersectionType(list(a))

        def union(*a: Type) -> UnionType:
            return UnionType(list(a))

        def arrow(self, *a: Type) -> CallableType:
            return fx.callable(self, *a)

        test_cases = [
            (intersect(omega, fx.a), fx.a),
            (intersect(fx.a, omega), fx.a),
            (intersect(fx.a, fx.b, fx.d, omega), intersect(fx.a, fx.b, fx.d)),
            (intersect(fx.a, fx.d), intersect(fx.a, fx.d)),
            (union(omega, fx.a), omega),
            (union(fx.a, omega), omega),
            (union(fx.a, fx.b, fx.d, omega), omega),
            (union(fx.a, fx.b, fx.d), union(fx.a, fx.b, fx.d)),
            (arrow(fx.a, omega), omega),
            (arrow(fx.a, fx.d), arrow(fx.a, fx.d)),
            (arrow(omega, fx.a), arrow(omega, fx.a))
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

    def assert_single_case(self, test_cases, current_test_case):
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
