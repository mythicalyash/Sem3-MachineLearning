# The Complete NumPy Guide — Beginner to Advanced

All code in this guide was tested against NumPy 2.4. Copy any block into a Python session and run it.

```python
import numpy as np   # the universal convention — always import it this way
```

---

## Table of Contents

**Part 1 — Foundations**
1. Why NumPy exists
2. The `ndarray`
3. Creating arrays
4. Array attributes
5. Data types (`dtype`)
6. Indexing and slicing
7. Arithmetic and broadcasting (intro)

**Part 2 — Intermediate**
8. Broadcasting rules, in depth
9. Reshaping
10. Joining and splitting
11. Boolean and fancy indexing
12. Universal functions (ufuncs)
13. Aggregations and the `axis` argument
14. Sorting and searching
15. Random numbers (the modern `Generator` API)
16. Linear algebra basics

**Part 3 — Advanced**
17. Views, copies, and memory layout
18. Vectorization and performance
19. Structured arrays
20. Masked arrays
21. `einsum`
22. Advanced linear algebra (eigenvalues, SVD, solving systems)
23. Custom ufuncs
24. Memory-mapped arrays
25. Where to go next

---

# Part 1 — Foundations

## 1. Why NumPy exists

A Python list is a list of pointers to objects scattered around memory. That's flexible but slow for numeric work. A NumPy array (`ndarray`) is a single contiguous block of memory holding elements of **one fixed type**. This buys you two things:

- **Speed**: operations run as compiled C loops instead of Python bytecode, element by element.
- **Vectorization**: you write `a + b` instead of a `for` loop, and the looping happens in C.

```python
import numpy as np
import time

n = 2_000_000
py_list = list(range(n))
np_arr = np.arange(n)

start = time.time()
py_result = [x * 2 for x in py_list]
print("Python list:", time.time() - start, "sec")

start = time.time()
np_result = np_arr * 2
print("NumPy array:", time.time() - start, "sec")
```

On a typical machine the NumPy version is 20–50x faster. That gap is the entire reason NumPy exists.

## 2. The `ndarray`

Everything in NumPy revolves around one object: `numpy.ndarray`, an n-dimensional array.

```python
a = np.array([1, 2, 3])          # 1-D array — a "vector"
b = np.array([[1, 2], [3, 4]])   # 2-D array — a "matrix"
c = np.array([[[1], [2]], [[3], [4]]])  # 3-D array — a "tensor"

print(a.ndim, b.ndim, c.ndim)    # 1 2 3
```

Think of dimensions as: 1-D = list of numbers, 2-D = table (rows × columns), 3-D = stack of tables, and so on.

## 3. Creating arrays

```python
np.array([1, 2, 3])              # from a Python list
np.zeros((2, 3))                 # 2x3 array of zeros
np.ones((3, 3))                  # 3x3 array of ones
np.full((2, 2), 7)               # filled with a constant
np.eye(3)                        # 3x3 identity matrix
np.arange(0, 10, 2)              # like range(): [0 2 4 6 8]
np.linspace(0, 1, 5)             # 5 evenly spaced points from 0 to 1
np.empty((2, 2))                 # uninitialized memory — fast, but garbage values
np.zeros_like(b)                 # same shape/dtype as b, filled with 0
```

`arange` vs `linspace`: `arange` is stepped by increment (may miss the endpoint due to float rounding); `linspace` is stepped by count and always includes the endpoint. Prefer `linspace` when you care about the exact number of points.

## 4. Array attributes

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape     # (2, 3) — rows, columns
a.ndim      # 2 — number of dimensions
a.size      # 6 — total number of elements
a.dtype     # dtype('int64') — element type
a.itemsize  # 8 — bytes per element
a.nbytes    # 48 — total bytes (size * itemsize)
```

## 5. Data types (`dtype`)

Every array has exactly one dtype. Common ones: `int32`, `int64`, `float32`, `float64`, `bool`, `complex128`, `<U10` (fixed-width Unicode string).

```python
a = np.array([1, 2, 3], dtype=np.float32)   # force a type at creation
b = a.astype(np.int32)                       # convert (creates a new array)

np.array([1, 2, 3]).dtype        # int64 on most 64-bit systems
np.array([1.0, 2, 3]).dtype      # float64 — one float promotes the whole array
np.array([1, 2, "3"]).dtype      # <U21 — one string promotes everything to string!
```

**Rule of thumb**: mixing types in `np.array()` upcasts everything to the "widest" common type. This is a frequent source of silent bugs — always check `.dtype` when something behaves oddly.

Smaller dtypes (`float32`, `int16`) save memory and can be faster, but risk overflow/precision loss:

```python
x = np.array([200], dtype=np.int8)
x + 100          # array([44], dtype=int8) — silently overflowed! int8 caps at 127
```

## 6. Indexing and slicing

Like Python lists, but extended to multiple dimensions.

```python
a = np.arange(10)
a[0]          # 0
a[-1]         # 9
a[2:5]        # [2 3 4]
a[::2]        # [0 2 4 6 8]
a[::-1]       # reversed

m = np.arange(12).reshape(3, 4)
#  [[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]

m[1, 2]       # 6            — row 1, col 2
m[1]          # [4 5 6 7]    — entire row 1
m[:, 2]       # [2 6 10]     — entire column 2
m[0:2, 1:3]   # [[1 2],[5 6]] — sub-block
m[-1]         # last row
```

**Critical**: basic slices (using `:`) return **views**, not copies. Modifying a slice modifies the original array. See Part 3, Section 17 for the full explanation.

```python
sub = m[0:2, 0:2]
sub[0, 0] = 999
print(m[0, 0])   # 999 — the original changed too!
```

## 7. Arithmetic and broadcasting (intro)

Arithmetic on arrays is element-wise by default:

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b     # [11 22 33]
a * b     # [10 40 90]   — element-wise, NOT matrix multiplication
a ** 2    # [1 4 9]
a @ b     # 140          — this is the dot product / matrix multiplication
```

When shapes don't match exactly, NumPy tries to **broadcast** the smaller one:

```python
a = np.array([1, 2, 3])
a + 5              # [6 7 8]   — the scalar 5 is "broadcast" to every element
```

Full broadcasting rules are covered in Section 8.

---

# Part 2 — Intermediate

## 8. Broadcasting rules, in depth

Broadcasting lets NumPy operate on arrays of different (but compatible) shapes without copying data. The rule, comparing shapes **from the right**:

1. If the dimensions are equal, or
2. one of them is 1,

then they're compatible; the size-1 dimension gets "stretched" to match. If neither holds, it's an error.

```python
x = np.array([1, 2, 3])           # shape (3,)
y = np.array([[1], [2], [3]])     # shape (3, 1)

x + y
# shape (3,) and (3,1) -> broadcast to (3,3):
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]]
```

Why this is powerful — normalizing every row of a matrix without a loop:

```python
data = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
row_means = data.mean(axis=1, keepdims=True)  # shape (2, 1)
centered = data - row_means                   # (2,3) - (2,1) broadcasts fine
```

`keepdims=True` is the trick that keeps broadcasting working cleanly — it preserves the reduced axis as size 1 instead of dropping it.

Common broadcasting error, and how to read it:

```python
np.array([1, 2, 3]) + np.array([1, 2])
# ValueError: operands could not be broadcast together with shapes (3,) (2,)
```

Read shapes right-to-left: `3` vs `2` — neither equal nor 1 — incompatible.

## 9. Reshaping

```python
a = np.arange(12)
a.reshape(3, 4)      # 3 rows, 4 cols — must have same total size (12)
a.reshape(3, -1)      # -1 means "infer this dimension" -> becomes (3, 4)
a.reshape(-1)          # flatten to 1-D
a.flatten()            # 1-D, always returns a copy
a.ravel()               # 1-D, returns a view when possible (faster)

a.T                     # transpose (works for any ndim; reverses axis order)
np.arange(6).reshape(2,3).T.shape   # (3, 2)
```

`reshape` returns a view when the data is contiguous, otherwise a copy — you generally don't need to worry about which unless you're optimizing for memory (see Section 17).

## 10. Joining and splitting

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate([a, b])          # [1 2 3 4 5 6]
np.stack([a, b])                 # new axis: shape (2, 3)
np.vstack([a, b])                # stack vertically -> shape (2, 3)
np.hstack([a, b])                # stack horizontally -> shape (6,)

m = np.arange(12).reshape(3, 4)
np.split(m, 3)                    # 3 equal pieces along axis 0
np.hsplit(m, 2)                    # split into 2 pieces along columns
```

`concatenate` needs matching dimensions except along the join axis; `stack` creates a brand-new axis (like turning a list of arrays into one array).

## 11. Boolean and fancy indexing

Boolean indexing — select elements matching a condition:

```python
a = np.array([1, -2, 3, -4, 5])
mask = a > 0
a[mask]             # [1 3 5]
a[a > 0]             # same thing, written inline
a[a < 0] = 0        # conditional assignment — zero out negatives in place
```

Fancy indexing — select using an array of indices:

```python
a = np.array([10, 20, 30, 40, 50])
idx = np.array([0, 2, 4])
a[idx]               # [10 30 50]

m = np.arange(12).reshape(3, 4)
m[[0, 2], [1, 3]]    # picks (0,1) and (2,3) -> [1 11]  (paired, not a grid!)
m[[0, 2]]             # rows 0 and 2 (whole rows)
```

`np.where` is the vectorized if/else:

```python
a = np.array([1, -2, 3, -4])
np.where(a > 0, a, 0)          # [1 0 3 0] — keeps positives, zeroes the rest
np.where(a > 0)                 # (array([0, 2]),) — indices where condition is True
```

**Important**: unlike basic slicing, boolean and fancy indexing always return **copies**, never views.

## 12. Universal functions (ufuncs)

Ufuncs are the vectorized element-wise functions that make NumPy fast: `np.sqrt`, `np.exp`, `np.log`, `np.sin`, `np.abs`, `np.add`, `np.maximum`, etc.

```python
a = np.array([1, 4, 9, 16])
np.sqrt(a)          # [1. 2. 3. 4.]
np.exp(a)            # element-wise e^x
np.maximum(a, 10)    # element-wise max against 10 -> [10 10 10 16]

# ufuncs also have useful methods:
np.add.reduce([1,2,3,4])     # 10  — same as sum
np.multiply.accumulate([1,2,3,4])   # [1 2 6 24] — running product
```

The rule: **if you're writing a Python `for` loop over array elements to do math, there's almost certainly a ufunc that replaces it and runs 10-100x faster.**

## 13. Aggregations and the `axis` argument

```python
m = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)

m.sum()             # 21 — sums everything
m.sum(axis=0)       # [5 7 9]   — sum DOWN each column (collapses axis 0, the rows)
m.sum(axis=1)       # [6 15]     — sum ACROSS each row (collapses axis 1, the columns)

m.mean(), m.std(), m.var(), m.min(), m.max()
m.argmax()          # index of the max in the FLATTENED array
m.argmax(axis=1)     # index of the max within each row
```

The mental model for `axis`: **`axis=N` means "collapse dimension N"** — the result loses that dimension (unless `keepdims=True`). A lot of people initially guess axis backwards; if unsure, test on a small array with known values.

## 14. Sorting and searching

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

np.sort(a)             # sorted copy: [1 1 2 3 4 5 6 9]
a.sort()                 # sorts in place
np.argsort(a)            # indices that WOULD sort the array
np.searchsorted(np.array([1,3,5,7]), 4)   # 2 — insertion point to keep it sorted

m = np.array([[3,1],[2,4]])
np.sort(m, axis=1)       # sort each row independently
```

## 15. Random numbers (the modern `Generator` API)

NumPy's old `np.random.seed()` / `np.random.rand()` interface still works but is legacy. Use `default_rng` for new code — it's faster and statistically more robust:

```python
rng = np.random.default_rng(seed=42)     # seed makes results reproducible

rng.random(5)                              # 5 floats in [0, 1)
rng.integers(0, 10, size=5)                # 5 ints in [0, 10)
rng.normal(loc=0, scale=1, size=(3, 3))    # normal/Gaussian distribution
rng.choice([1, 2, 3, 4, 5], size=3, replace=False)   # random sample, no repeats
rng.shuffle(a)                              # shuffles `a` in place
```

Always pass a seed when you need reproducible results (tests, tutorials, debugging).

## 16. Linear algebra basics

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B                    # matrix multiplication (preferred syntax)
np.dot(A, B)               # equivalent to @  for 2-D arrays
A.T                          # transpose
np.linalg.inv(A)             # matrix inverse
np.linalg.det(A)             # determinant
np.trace(A)                  # sum of diagonal elements
```

---

# Part 3 — Advanced

## 17. Views, copies, and memory layout

This is the section that separates "using NumPy" from "understanding NumPy."

An array is a thin wrapper around a raw memory buffer plus metadata: `shape`, `dtype`, and **strides** — the number of bytes to step to move to the next element along each axis.

```python
a = np.arange(24).reshape(4, 6)
a.strides       # (48, 8) — jump 48 bytes for the next row, 8 bytes for the next column
a.dtype.itemsize  # 8 bytes per int64 element (6 cols * 8 bytes = 48, matches row stride)
```

This is *why* operations like `reshape`, basic slicing, and `.T` are nearly free — they just create a new view with different shape/strides over the *same* underlying memory, without copying a single byte.

```python
a = np.arange(6)
b = a.reshape(2, 3)
np.shares_memory(a, b)     # True — b is a view over a's memory
b[0, 0] = 999
a[0]                          # 999 — a changed too
```

The reliable way to check whether two arrays share memory is `np.shares_memory(x, y)` — checking `y.base is x` can be misleading, because views can chain through multiple layers (e.g. `a[1:3, 2:4].base` may point to `a`'s own base rather than `a` itself, even though they still share memory).

**Which operations return views vs copies:**

| Operation | Result |
|---|---|
| Basic slicing (`a[1:3]`, `a[:, 0]`) | View |
| `.reshape()` (when data is contiguous) | View |
| `.T`, `.transpose()` | View |
| `.ravel()` | View when possible, else copy |
| Boolean indexing (`a[a > 0]`) | Copy |
| Fancy indexing (`a[[0, 2, 4]]`) | Copy |
| `.flatten()` | Always copy |
| Arithmetic (`a + b`) | Always a new array |
| `.copy()` | Explicit, forced copy |

If you need to guarantee independence from the original, call `.copy()` explicitly.

**C order vs Fortran order** — the two ways to lay out a multi-dimensional array in flat memory:

```python
a = np.arange(6).reshape(2, 3, order='C')   # row-major (default): rows are contiguous
b = np.arange(6).reshape(2, 3, order='F')   # column-major: columns are contiguous
a.flags['C_CONTIGUOUS'], b.flags['F_CONTIGUOUS']   # True, True
```

This matters for performance: looping over the *contiguous* axis is cache-friendly; looping over the other axis causes scattered memory access and is slower on large arrays.

## 18. Vectorization and performance

The single biggest performance lesson in NumPy: **avoid Python-level loops over array elements.**

```python
# Slow — Python loop, ~1-2 seconds for a million elements
a = np.arange(1_000_000)
result = np.empty_like(a)
for i in range(len(a)):
    result[i] = a[i] ** 2 + 1

# Fast — vectorized, milliseconds
result = a ** 2 + 1
```

Other performance techniques:

```python
# Preallocate with `out=` instead of creating new arrays in a loop
result = np.empty(1000)
np.add(a, b, out=result)          # writes directly into `result`, no new allocation

# Use in-place operators to avoid allocating a fresh array
a += 1        # in-place, no new array
a = a + 1     # allocates a new array

# np.einsum can fuse multiple operations into a single fast pass (Section 21)
```

For loops that genuinely can't be vectorized (complex conditional logic per element, sequential dependencies), the next step up is **Numba** (`@njit` decorator, JIT-compiles Python to machine code) or **Cython** — both let you keep loop-based logic but run at C speed. That's a good next stop once you outgrow pure NumPy vectorization.

## 19. Structured arrays

For heterogeneous, table-like data with named fields — a lightweight alternative to pandas when you don't need a full DataFrame:

```python
people = np.array(
    [("Alice", 30, 55.5), ("Bob", 25, 70.2)],
    dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")]
)

people["name"]           # array(['Alice', 'Bob'], dtype='<U10')
people["age"].mean()      # 27.5
people[0]                  # ('Alice', 30, 55.5) — one full record
people[people["age"] > 26]  # boolean indexing works normally, filters records
```

## 20. Masked arrays

For data with missing or invalid values you want to exclude from calculations without deleting them:

```python
import numpy.ma as ma

data = np.array([1, 2, -999, 4, 5])       # -999 is a "missing value" sentinel
masked = ma.masked_equal(data, -999)

masked.mean()          # 3.0 — the -999 is excluded automatically
masked.filled(0)        # replace masked entries with 0 for output: [1 2 0 4 5]
```

## 21. `einsum`

Einstein summation notation — a single expressive syntax for a huge family of matrix/tensor operations, and often faster than chaining multiple NumPy calls because it fuses them into one pass.

```python
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)

np.einsum('ij,jk->ik', A, B)     # equivalent to A @ B
np.einsum('ii', np.eye(3))         # trace (sum of diagonal): 3.0
np.einsum('ij->ji', A)              # transpose
np.einsum('ij,ij->i', A, A)        # row-wise dot product (sum of squares per row)
np.einsum('i,i->', A[0], A[0])     # dot product of a single row with itself
```

Reading the notation: letters label axes; repeated letters across inputs mean "multiply and sum over that axis"; the `->` output side lists which axes survive. It takes practice, but once it clicks it replaces a lot of fragile `reshape`/`sum`/`transpose` chains with one readable line.

## 22. Advanced linear algebra

```python
A = np.array([[4, 2], [1, 3]])

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)
eigvals            # [2. 5.]  — the eigenvalues
eigvecs             # corresponding eigenvectors as columns

# Singular Value Decomposition — the workhorse of PCA, compression, recommender systems
M = np.array([[1, 2], [3, 4]])
U, S, Vt = np.linalg.svd(M)      # M == U @ diag(S) @ Vt

# Solving linear systems (Ax = b) — prefer this over computing inv(A) @ b,
# it's both faster and numerically more stable
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)          # solves Ax = b directly

# Norms
np.linalg.norm(np.array([3, 4]))            # 5.0 — Euclidean (L2) norm
np.linalg.norm(np.array([3, 4]), ord=1)     # 7.0 — L1 (Manhattan) norm
```

## 23. Custom ufuncs

When you have a scalar Python function and want it to behave like a NumPy ufunc (broadcast automatically over arrays):

```python
def piecewise(x):
    return x**2 + 1 if x > 0 else 0

vec_fn = np.vectorize(piecewise)
vec_fn(np.array([-2, -1, 0, 1, 2]))    # [0 0 0 2 5]
```

Fair warning: `np.vectorize` is a convenience wrapper, not a performance tool — it still calls your Python function once per element under the hood, so it's not faster than a plain loop. For real speed with custom per-element logic, reach for Numba's `@vectorize`/`@njit` decorators instead, which actually JIT-compile the function.

## 24. Memory-mapped arrays

For datasets too large to fit in RAM, `np.memmap` lets you treat a file on disk as if it were a NumPy array, loading only the parts you touch:

```python
# Create and write to a memory-mapped file
arr = np.memmap('data.dat', dtype='float32', mode='w+', shape=(1000, 1000))
arr[:] = np.random.rand(1000, 1000)
arr.flush()          # ensure it's written to disk

# Later, reopen without loading the whole thing into RAM
arr2 = np.memmap('data.dat', dtype='float32', mode='r', shape=(1000, 1000))
arr2[500, 500]        # only this region gets read from disk
```

## 25. Where to go next

Once this feels comfortable, natural next steps:

- **pandas** — built on NumPy, adds labeled rows/columns and heterogeneous data handling.
- **Numba** — JIT-compile your own Python loops to machine-code speed with a decorator.
- **SciPy** — adds optimization, statistics, signal processing, sparse matrices on top of NumPy.
- **CuPy** — a near drop-in NumPy replacement that runs on GPU.
- **PyTorch/JAX arrays** — if you're heading toward machine learning, their array APIs are deliberately close to NumPy's, so everything here transfers directly.

A good way to consolidate this: pick a small dataset and re-implement something you'd normally do with loops (e.g. a moving average, a distance matrix between points, a simple image filter) entirely with vectorized NumPy — no `for` loops over elements at all.
