"""
Utitility functions for experiments in the HyQu Group at Hönggerberg.
Based on the RWA formalism presented in Xinyu's master thesis.
"""

__all__ = [    
    'simplify_exp',
    'conjugate_expression',    
    'drop_time_dependence',  
    'simplify_recursive_commutator',
    'bch_expansion',
    'unitary_transformation',
    'hamiltonian_transformation',
    'time_integral',
    'trace',
    'time_average'
    ]

import warnings
from collections import namedtuple
from sympy import (Add, Mul, Pow, exp, latex, Integral, Sum, Integer, Rational, Symbol,
                   I, pi, simplify, oo, DiracDelta, KroneckerDelta, collect,
                   factorial, diff, Function, Derivative, Eq, symbols,
                   Matrix, Equality, MatMul, Dummy, conjugate)

from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy import (sin, cos, sinh, cosh)
from sympsi import Operator, Commutator, Dagger
from sympsi.operatorordering import normal_ordered_form
from sympsi.expectation import Expectation
from sympsi.pauli import (SigmaX, SigmaY, SigmaMinus, SigmaPlus)
from sympsi.boson import (BosonOp, MultiBosonOp)
from sympsi.qutility import (drop_terms_containing, extract_operators, extract_operator_products, split_coeff_operator, qsimplify, subs_single)

debug = False
t = symbols("t")
# Todo: Make this generic so other symbols can be used for the qubit operators etc.
a,g1, g2, Om1, Om2, t, Hsym = symbols("alpha, g_1, g_2, Omega_1, Omega_2, t, H")
wq, wk1, wk2, wj1, wj2 = symbols("omega_q, omega_k1, omega_k2, omega_j1, omega_j2")

mq = MultiBosonOp("m", 0, 'discrete')
m1 = MultiBosonOp("m", 1, 'discrete')
m2 = MultiBosonOp("m", 2, 'discrete')

# -----------------------------------------------------------------------------
# Simplification of expressions
#
# The are two new helper functions here: simplify_exp and conjugate_expression. Their purpose is explained below.

def _help_exp(e):
    # Important: This function is currently only applicable to summands of the form on p. 6 in Xinyu's thesis.
    # This means it does not prevent operators and exponentials with operators from being reversed in order!
    new = 1
    h = 0
    factors = 1
    for arg in e.args:
        if isinstance(arg, exp):
            h += arg.exp
        elif isinstance(arg, Pow):
            p = arg.args[-1]  # This stores the power 
            arg = simplify(arg.args[0]) # This stores the expression to be "powered".
            if p < 0: # This deals with expressions of form 1 / (...)
                f = 0
                for a in arg.args: # We don't want this iteration to include the power arg at the end
                    if isinstance(a, exp):
                        h -= a.exp # The minus takes the 1/ into account
                    else:
                        f += a
                factors /= f
            else: # This else is a bit redundant with the else below, but we need to catch "normal" power of form x^2
                factors *= arg**p
        else: 
            factors *= arg
    new *= exp(simplify(h))
    new *= factors
    return new
    
def simplify_exp(e, sub_list):
    '''Contracts scalar exponential products in an expression 
    and substitutes relevant terms in the resulting exponentials'''
    s = 0
    if isinstance(e, Add):
        terms = e.args
        for term in terms:
            s += _help_exp(term)
    elif isinstance(e, Mul):
        s = _help_exp(e)
        
    for tuple in sub_list:
        s = s.subs(tuple[0], tuple[1])
    return s
    
# Note that separate if-cases are necessary for the many different types of symbols in sympy. The cases might still be incomplete.

def _conjugate_Mul(e):
    new = 1
    for arg in e.args:
        if isinstance(arg, Rational) or arg==t:
            new *= arg
            continue
        if arg == I:
            new *= -I
        if isinstance(arg, Symbol):
            new *= conjugate(arg)
        if isinstance(arg, conjugate):
            new *= conjugate(arg)
        if isinstance(arg, Pow):
            new *= conjugate(arg.args[0])**arg.args[1]
        if isinstance(arg, BosonOp):
            new *= Dagger(arg)
        if isinstance(arg, Add):
            new *= _conjugate_Add(arg)
        if isinstance(arg, Mul):
            new *= _conjugate_Mul(arg)
        if isinstance(arg, exp):
            new *= _conjugate_exp(arg)
    return new

def _conjugate_Add(e):
    new = 0
    for arg in e.args:
        if isinstance(arg, Rational):
            new += arg
        if arg == I:
            new += -I
        if isinstance(arg, Symbol):
            new += conjugate(arg)
        if isinstance(arg, conjugate):
            new += conjugate(arg)
        if isinstance(arg, Pow):
            new += conjugate(arg.args[0])**arg.args[1] # Example: For arg = x**2, arg.args would be (x, 2)
        if isinstance(arg, BosonOp):
            new += Dagger(arg)
        if isinstance(arg, Add):
            new += _conjugate_Add(arg)
        if isinstance(arg, Mul):
            new += _conjugate_Mul(arg)
        if isinstance(arg, exp):
            new += _conjugate_exp(arg)
    return new

def _conjugate_exp(e):
    if isinstance(e.exp, Add):
        return exp(_conjugate_Add(e.exp))
    if isinstance(e.exp, Mul):
        new = _conjugate_Mul(e.exp)
        return exp(simplify(new.expand()))
    
def conjugate_expression(e, sub):   
    '''A function for complicated expressions involving both scalars and operators. 
       Scalars are conjugated, operators are daggered.'''
    sub_list = []
    for w in sub:
        sub_list.append((conjugate(w), w)) #We might want some quantities to be real -> hence introduce sub_list
    if isinstance(e, Add):
        return _conjugate_Add(e).subs(sub_list)
    if isinstance(e, Mul):
        return _conjugate_Mul(e).subs(sub_list)
    if isinstance(e, exp):
        return _conjugate_exp(e).subs(sub_list)
    else:
        return None

# -----------------------------------------------------------------------------
# Utility functions for manipulating operator expressions
#
# I haven't actually used this function, but I left it anyhow.

def drop_time_dependence(e):
    """
    Drop terms contaning time-dependent coeffcients
    """
    cno = split_coeff_operator(e)
    for tuple in cno:
        c = tuple[0]
    if t in c.free_symbols:
        e = e.subs(c,0)
    return e


# -----------------------------------------------------------------------------
# Commutators and BCH expansions
#
# The differences to the functions in qutility.py are:
# 1) The extra function "simplify_recursive_commutator" which properly carries out all commutators via .expand(commutator=True)
#    The function _bch_expansion is responsible for calling either recursive_commutator or simplify_recursive_commutator, 
#    based on the bool variable simplify_commutator (see next subsection on Transformation functions)
# 2) The function _expansion_search includes a quick fix for recognizing series expansions with two free symbols.

def recursive_commutator(a, b, n=1):
    """
    Generate a recursive commutator of order n:

    [a, b]_1 = [a, b]
    [a, b]_2 = [a, [a, b]]
    [a, b]_3 = [a, [a, b]_2] = [a, [a, [a, b]]]
    ...

    """
    if n == 1:
        return Commutator(a, b)
    else:
        return Commutator(a, recursive_commutator(a, b, n-1))
    
def simplify_recursive_commutator(a, b, n=1):
    """
    Generate a recursive commutator of order n:

    [a, b]_1 = [a, b]
    [a, b]_2 = [a, [a, b]]
    [a, b]_3 = [a, [a, b]_2] = [a, [a, [a, b]]]
    ...

    """
    if n == 1:
        return Commutator(a, b).expand(commutator=True).expand(commutator=True).doit()
    else:
        return Commutator(a, simplify_recursive_commutator(a, b, n-1)).expand(commutator=True).expand(commutator=True).doit() 
        
        
def _bch_expansion(A, B, N=10, simplify_commutator=True):
    """
    Baker–Campbell–Hausdorff formula:

    e^{A} B e^{-A} = B + 1/(1!)[A, B] +
                     1/(2!)[A, [A, B]] + 1/(3!)[A, [A, [A, B]]] + ...
                   = B + Sum_n^N 1/(n!)[A, B]^n

    Truncate the sum at N terms.
    """
    e = B
    if simplify_commutator==True:
        for n in range(1, N):
            e += simplify_recursive_commutator(A, B, n=n) / factorial(n)
    else:
        for n in range(1, N):
            e += recursive_commutator(A, B, n=n) / factorial(n)

    return e


def _order(e):
    fs = list(e.free_symbols)
    if isinstance(e, Pow) and e.base == fs[0]:
        return e.exp
    elif isinstance(e, Mul):
        o = sum([_order(arg) for arg in e.args])
        return o
    elif isinstance(e, Add):
        o = max([_order(arg) for arg in e.args])
        return o
    elif e.is_Symbol:
        return 1
    else:
        return 0
    
def _lowest_order_term(e):

    if isinstance(e, Add):
        min_order = _order(e.args[0])
        min_expr = e.args[0]
        for arg in e.args:
            arg_order  = _order(arg)
            if arg_order < min_order:
                min_order = arg_order
                min_expr = arg
        return min_expr, min_order
    else:
        return e, _order(e)


def _expansion_search(e, rep_list, N):
    """
    Search for and substitute terms that match a series expansion of
    fundamental math functions.

    e: expression

    rep_list: list containing dummy variables

    """
    if e.find(I):
        is_complex = True
    else:
        is_complex = False

    if debug:
        print("_expansion_search: ", e)

    try:
        dummy = Dummy()
        # The 0, 1 and 2 in flist... refer to the lowest order of x
        flist0 = [exp, lambda x: exp(-x), cos, cosh]

        flist1 = [lambda x: (exp(x) - 1) / x,
                  lambda x: (1 - exp(-x)) / x,
                  lambda x: sin(x) / x,
                  lambda x: sinh(x) / x]

        flist2 = [lambda x: (1 - cos(x))/(x**2/2),
                  lambda x: (cosh(x)-1)/(x**2/2)]

        if is_complex:
            iflist0 = [lambda x: exp(I*x),
                       lambda x: exp(-I*x)]
            iflist1 = [lambda x: (exp(I*x) - 1) / (I*x),
                       lambda x: (1 - exp(-I*x)) / (I*x)]

            flist0 = iflist0 + flist0
            flist1 = iflist1 + flist1

        flist = [flist0, flist1, flist2]
        fseries = {}

        if isinstance(e, Mul):
            e_args = [e]
        elif isinstance(e, Add):
            e_args = e.args
        else:
            return e

        newargs = []
        for e in e_args:
            if isinstance(e, Mul):
                c, nc = e.args_cnc()
                if nc and c:
                    c_expr = Mul(*c).expand()
                    d, d_order = _lowest_order_term(c_expr)
                    c_expr_normal = (c_expr / d).expand()
                    c_expr_subs = c_expr_normal
                    free = list(c_expr_subs.free_symbols) 
                    #Together with the if clause below, this is a quick fix for doing the substitution properly

                    for alpha in rep_list:
                        if alpha not in c_expr_subs.free_symbols:
                            continue
                        if len(free)==2: # For example, alpha and beta would be t and omega (frequency).
                            # This is necessary since we want both symbols to be included in the search for a series 
                            # expansion below (in c_expr_subs.subs(fseries[f]...))
                            alpha = free[0]
                            beta = free[1]

                        for f in flist[d_order]: # Now we go through the function candidates with the right order
                            if f not in fseries.keys():
                                fseries[f] = f(dummy).series(dummy, n=N-d_order).removeO()
                            c_expr_subs = c_expr_subs.subs(fseries[f].subs(dummy, alpha*beta), f(alpha*beta))
                            if c_expr_subs != c_expr_normal:
                                break
                        if c_expr_subs != c_expr_normal:
                            break
                    newargs.append(d * c_expr_subs * Mul(*nc))
                else:
                    newargs.append(e)
            else:
                newargs.append(e)

        return Add(*newargs)

    except Exception as e:
        print("Failed to identify series expansions: " + str(e))
        return e


def bch_expansion(A, B, N=6, collect_operators=None, independent=False,
                  expansion_search=True, simplify_commutator=True):

    # Use BCH expansion of order N

    if debug:
        print("bch_expansion: ", A, B)

    cno = split_coeff_operator(A)
    if isinstance(cno, list):
        nvar = len(cno)
        c_list = []
        o_list = []
        for n in range(nvar):
            c_list.append(cno[n][0])
            o_list.append(cno[n][1])
    else:
        nvar = 1
        c_list, o_list = [cno[0]], [cno[1]] # c_list stores coefficients, o_list stores operators

    if debug:
        print("A coefficient: ", c_list)

    rep_list = [] # This list will store one Dummy variable for each symbol in var_list.
                  # Below, these Dummies are substituted into the expression A_rep, which gets passed to _bch_expansion.
                  # Finally, the original symbols get re-substituted in e_collected at the end.
    var_list = []

    for n in range(nvar):
        rep_list.append(Dummy())

        coeff, sym = c_list[n].as_coeff_Mul()
        if isinstance(sym, Mul):
            sym_ = simplify(sym)
            if I in sym_.args:
                var_list.append(sym_/I)
            elif any([isinstance(arg, exp) for arg in sym_.args]):
                nexps = Mul(*[arg for arg in sym_.args
                              if not isinstance(arg, exp)])
                exps = Mul(*[arg for arg in sym_.args if isinstance(arg, exp)])

                if I in simplify(exps).exp.args:
                    var_list.append(nexps)
                else:
                    var_list.append(sym_)
        else:
            var_list.append(sym)

    A_rep = A.subs({var_list[n]: rep_list[n] for n in range(nvar)})

    e_bch_rep = _bch_expansion(A_rep, B, N=N, simplify_commutator=simplify_commutator).doit(independent=independent)

    if debug:
        print("simplify: ")

    e = qsimplify(normal_ordered_form(e_bch_rep.expand(),
                                      recursive_limit=25,
                                      independent=independent).expand())
    if debug:
        print("extract operators: ")

    ops = extract_operator_products(e, independent=independent)

    ops = list(reversed(sorted(ops, key=lambda x: len(str(x)))))

    if debug:
        print("operators in expression: ", ops)

    if collect_operators:
        e_collected = collect(e, collect_operators)
    else:
        e_collected = collect(e, ops) 

    if debug:
        print("search for series expansions: ", expansion_search)

    if debug:
        print("e_collected: ", e_collected)

    if expansion_search and c_list:
        e_collected = _expansion_search(e_collected, rep_list, N) # This is a crucial step where the finite series expansion                                                                     # must be recognized. It sometimes still goes wrong.
        e_collected = e_collected.subs({rep_list[n]: var_list[n]
                                        for n in range(nvar)})

        return e_collected
    else:
        return e_collected.subs(
            {rep_list[n]: var_list[n] for n in range(nvar)})
    


# -----------------------------------------------------------------------------
# Transformations
#
# The difference to the unitary_transformation and hamiltonian_transformation functions in qutility.py is the extra
# bool variable simplify_commutator which gets passed to bch_expansion.

def unitary_transformation(U, O, N=6, collect_operators=None,
                           independent=False, allinone=False,
                           expansion_search=True, simplify_commutator=True):
    """
    Perform a unitary transformation

        O = U O U^\dagger

    and automatically try to identify series expansions in the resulting
    operator expression.
    """
    if not isinstance(U, exp):
        raise ValueError("U must be a unitary operator on the form "
                         "U = exp(A)")
    if simplify_commutator==True:
        print("using simplified commutator")

    A = U.exp

    if debug:
        print("unitary_transformation: using A = ", A)

    if allinone:
        return bch_expansion(A, O, N=N, collect_operators=collect_operators,
                             independent=independent,
                             expansion_search=expansion_search, simplify_commutator=simplify_commutator)
    else:
        ops = extract_operators(O.expand())
        ops_subs = {op: bch_expansion(A, op, N=N,
                                      collect_operators=collect_operators,
                                      independent=independent,
                                      expansion_search=expansion_search, simplify_commutator=simplify_commutator)
                    for op in ops}

        #return O.subs(ops_subs, simultaneous=True) # XXX: this this
        return subs_single(O, ops_subs)


def hamiltonian_transformation(U, H, N=6, collect_operators=None,
                               independent=False, expansion_search=True, simplify_commutator=True):
    """
    Apply an unitary basis transformation to the Hamiltonian H:

        H = U H U^\dagger -i U d/dt(U^\dagger)

    """
    t = [s for s in U.exp.free_symbols if str(s) == 't']
    if t:
        t = t[0]
        H_td = - I * U * diff(exp(-U.exp), t)
    else:
        H_td = 0

    # H_td = I * diff(U, t) * exp(- U.exp)  # hack: Dagger(U) = exp(-U.exp)
    H_st = unitary_transformation(U, H, N=N,
                                  collect_operators=collect_operators,
                                  independent=independent,
                                  expansion_search=expansion_search, simplify_commutator=simplify_commutator)
    return H_st + H_td

# -----------------------------------------------------------------------------
# Tools for RWA
#
#

def _help_integral(H):
    H_int = H
    for arg in H.args:
        if isinstance(arg, exp):
            e = simplify(arg.exp.subs([(Dagger(mq), 1), (mq, 1)])) / (I*t)
            H_int /= e
    return H_int
    
def time_integral(H):
    # This function takes a Hamiltonian whose summands each have a single exponential and "integrates",
    # i.e. divides by the expression in the exponential. This will result in operators in denominators. 
    # We accept this on the pragmatic basis that a partial trace will be taken later on.
    H_int = 0
    if isinstance(H, Add):
        for arg in H.args:
            H_int += _help_integral(arg)
    else: # In case H consists of a single summand
        H_int = _help_integral(H)
    return H_int

def time_average(H, resonance_list):
    # Note: Resonance_list should contain expressions of the form "delta_k1 + delta_k2 - delta_j1 - delta_j2"
    #H = simplify_exp(simplify(simplify_exp(H, [])), []) # Looks ugly, but does the trick for now. 
    H = simplify_exp(H, [])
    H_avg = 0
    for s in H.args:
        if t not in s.free_symbols:
            H_avg += s # This adds diagonal terms that don't contain 't' at all. 
        else:          # Otherwise, we check whether the resonance condition is fulfilled
            r = 0
            f = 1      # f collects the multiplicative factors and operators without the exponential
            for e in s.args:
                if isinstance(e, exp):
                    # The fact that the SWAP term has no alpha is no problem. The point here is to extract a potentially
                    # resonant exponential. The SWAP term is simply already in the correct form :)
                    r = simplify(drop_terms_containing(e.exp.expand(), [a])/(I*t))
                else: 
                    f *= e
            if r in resonance_list:
                H_avg += f 
    return H_avg



def _check_operator(e):
    if e == mq:
        return False
    else: # includes case e == Dagger(mq)
        return True

def trace(e):
    # The expression e could be quite complicated, arising from the product of Hamiltonians in higher order RWAs.
    e_h = 0
    for s in e.args: # Working on assumption that e really is a sum, not just one summand.
        alpha = False
        keep = True
        if isinstance(s.args[-1], exp):
            keep = _check_operator(s.args[-2]) # checks whether left of exp we have mq 
        else:
            alpha = _check_operator(s.args[-1])
            
        if keep == True:
            s_h = 1
            for arg in s.args:
                if isinstance(arg, (Symbol, Rational, conjugate)):
                    s_h *= arg 
                if isinstance(arg, Pow) and arg.args[-1] > 0:
                    s_h *= arg
                if isinstance(arg, Pow) and arg.args[-1] < 0: # This case is for the lovely 1/Operator expressions :)
                    if alpha == False: #This was the case when the |0> was not raised.
                        h = 0
                        for term in arg.args[0].args: # Unfortunately, there is a problem with the qutility function drop_terms_containing
                            if term != a and term!= -1*a:
                                h += term
                        s_h /= h
                    if alpha == True: #This was the case when a right-most Dagger(q) raised the |0> ket to |1>
                        s_h *= arg.subs([(Dagger(mq), 1), (mq, 1)])
                if isinstance(arg, BosonOp):
                    if arg != mq and arg != Dagger(mq): # This is currently implemented in good faith that the remaining expressions are always
                        # in the form qq*, qq*qq* as in Xinyu's thesis on p. 31
                        s_h *= arg
                if isinstance(arg, exp):
                    s_h *= exp(arg.exp.subs([(Dagger(mq), 1), (mq, 1)]))
            e_h += s_h
    return simplify_exp(e_h, [])


def time_average(H, resonance_list):
    # Note: Resonance_list should contain expressions of the form "delta_k1 + delta_k2 - delta_j1 - delta_j2"
    # Currently, time_average only works if H is a real sum (not just one summand)
    
    #H = simplify_exp(H, []) 
    H_avg = 0
    for s in H.args:
        if t not in s.free_symbols:
            H_avg += s # This adds diagonal terms that don't contain 't' at all. 
        else:          # Otherwise, we check whether the resonance condition is fulfilled
            r = 0
            f = 1      # f collects the multiplicative factors and operators without the exponential
            for e in s.args:
                #want = True
                if isinstance(e, exp): 
                    h = e.exp.expand()
                    for term in h.args:
                        if a not in term.free_symbols:
                            r += term
                    r = simplify(r)/(I*t)
                else: 
                    f *= e
            if r in resonance_list or -1*r in resonance_list or 2*r in resonance_list or -2*r in resonance_list:
                H_avg += f 
    return H_avg