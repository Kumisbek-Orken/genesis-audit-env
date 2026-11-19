"""
Safe arithmetic calculator based on the ``ast`` module.

Allows only decimal numbers, parentheses, and the binary/unary operators
+, -, *, /.  All arithmetic is performed with ``decimal.Decimal`` for
exactness.

Rejects any variables, function calls, attribute access, exponentiation
with large exponents, and all other Python constructs.  Raises
``CalculatorError`` on any violation.
"""

from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation, DivisionByZero
from typing import Union


class CalculatorError(Exception):
    """Raised when the calculator input is invalid or unsafe."""


# Maximum length of the raw expression string (generous but bounded).
_MAX_EXPR_LEN = 2048

# Allowed AST node types.
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.Constant,       # numeric literals  (Python 3.8+)
    ast.UnaryOp,
    ast.BinOp,
    ast.UAdd,
    ast.USub,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
)


def _validate_node(node: ast.AST) -> None:
    """Recursively validate every AST node is in the allow-list."""
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        raise CalculatorError(
            f"Disallowed expression element: {type(node).__name__}. "
            "Only numbers, +, -, *, / and parentheses are allowed."
        )

    if isinstance(node, ast.Constant):
        # Only int and float literals (we convert to Decimal later).
        if not isinstance(node.value, (int, float)):
            raise CalculatorError(
                f"Only numeric literals are allowed, got {type(node.value).__name__}: "
                f"{node.value!r}"
            )
        # Guard against huge float exponents smuggled through literals like 1e9999.
        if isinstance(node.value, float) and (
            node.value != node.value  # NaN
            or abs(node.value) == float("inf")
        ):
            raise CalculatorError(
                f"Non-finite numeric literal: {node.value!r}"
            )

    # Recurse into child nodes.
    for child in ast.iter_child_nodes(node):
        _validate_node(child)


def _evaluate(node: ast.AST) -> Decimal:
    """Recursively evaluate a validated AST tree using Decimal arithmetic."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        try:
            return Decimal(str(node.value))
        except InvalidOperation:
            raise CalculatorError(f"Cannot convert {node.value!r} to Decimal")

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand

    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            try:
                return left / right
            except (DivisionByZero, InvalidOperation):
                raise CalculatorError("Division by zero")

    # Should never get here after validation.
    raise CalculatorError(f"Unexpected node: {type(node).__name__}")  # pragma: no cover


def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic *expression* and return the result as a string.

    Parameters
    ----------
    expression:
        A string containing only decimal numbers, +, -, *, / and
        parentheses.

    Returns
    -------
    str
        The exact decimal result as a string.

    Raises
    ------
    CalculatorError
        On any parse error, disallowed construct, or arithmetic error.

    Examples
    --------
    >>> calculate("2 + 2")
    '4'
    >>> calculate("100.50 * 3")
    '301.50'
    """
    if not isinstance(expression, str):
        raise CalculatorError("expression must be a string")
    expression = expression.strip()
    if not expression:
        raise CalculatorError("expression must not be empty")
    if len(expression) > _MAX_EXPR_LEN:
        raise CalculatorError(
            f"expression too long ({len(expression)} chars, max {_MAX_EXPR_LEN})"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Syntax error: {exc}") from exc

    _validate_node(tree)
    result = _evaluate(tree)

    # Normalize: remove trailing zeros but keep at least '0'.
    normalized = result.normalize()
    return str(normalized)
