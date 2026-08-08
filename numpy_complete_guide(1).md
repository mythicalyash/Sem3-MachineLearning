# The Complete NumPy Reference Notes — Beginner to Advanced

Every function below is documented in the same format and every code example has been run against NumPy 2.4:

> ### `signature`
> **Definition:** what it does, in plain language.
> **Parameters:** each argument, its type, and what it controls.
> **Returns:** what you get back.
> ```python
> example usage
> ```
> **Note:** gotchas, performance notes, or common mistakes (only where relevant).

```python
import numpy as np   # universal convention
```

---

## Table of Contents

**Part 1 — Foundations:** array creation, attributes, dtypes, indexing
**Part 2 — Intermediate:** broadcasting, reshaping, joining/splitting, indexing, ufuncs, aggregation, sorting, random, linear algebra
**Part 3 — Advanced:** memory internals, performance, structured/masked arrays, einsum, advanced linear algebra, custom ufuncs, memory-mapping, utility functions

---

# Part 1 — Foundations

## 1. Why NumPy exists

A Python `list` stores pointers to scattered objects; a NumPy `ndarray` stores raw numbers contiguously in memory with one fixed type. That lets math run as compiled C loops instead of Python bytecode — typically 20-100x faster for numeric work, and the reason every scientific Python library builds on it.

## 2. Array creation functions

> ### `np.array(object, dtype=None, copy=True, ndmin=0)`
> **Definition:** Builds an `ndarray` from a Python list, tuple, or nested sequence.
> **Parameters:**
> - `object` — the input data (list, tuple, nested sequence, or another array).
> - `dtype` — force a specific element type (e.g. `np.float32`). If omitted, NumPy infers the "widest" type needed to hold all elements.
> - `copy` — if `True` (default), always copies the input data; if `False`, reuses the buffer when possible.
> - `ndmin` — forces the result to have at least this many dimensions.
> **Returns:** a new `ndarray`.
> ```python
> np.array([1, 2, 3])                    # array([1, 2, 3])
> np.array([1, 2, 3], dtype=np.float32)  # array([1., 2., 3.], dtype=float32)
> np.array([[1, 2], [3, 4]])              # 2-D array
> ```
> **Note:** mixing types upcasts everything — `np.array([1, 2, "3"])` becomes an all-string array. Always check `.dtype` if a result looks wrong.

> ### `np.zeros(shape, dtype=float)`
> **Definition:** Creates an array filled entirely with `0`.
> **Parameters:** `shape` — int or tuple of ints, e.g. `(3, 4)`. `dtype` — element type, default `float64`.
> **Returns:** new array of the given shape, all zeros.
> ```python
> np.zeros((2, 3))    # 2x3 array of 0.0
> np.zeros(5, dtype=int)   # [0 0 0 0 0]
> ```

> ### `np.ones(shape, dtype=float)`
> **Definition:** Same as `np.zeros` but fills with `1` instead of `0`. Same parameters and behavior.
> ```python
> np.ones((3, 3))
> ```

> ### `np.full(shape, fill_value, dtype=None)`
> **Definition:** Creates an array of the given shape filled with a constant value of your choosing.
> **Parameters:** `shape`, `fill_value` — the constant to fill with, `dtype` — inferred from `fill_value` if omitted.
> ```python
> np.full((2, 2), 7)     # [[7 7],[7 7]]
> ```

> ### `np.eye(N, M=None, k=0, dtype=float)`
> **Definition:** Creates a 2-D array with `1`s on a diagonal and `0`s elsewhere.
> **Parameters:** `N` — number of rows. `M` — number of columns (defaults to `N`, giving a square matrix). `k` — index of the diagonal to fill (0 = main diagonal, positive = above, negative = below).
> **Returns:** identity-like matrix.
> ```python
> np.eye(3)          # 3x3 identity matrix
> np.eye(3, k=1)      # 1s on the diagonal just above the main one
> ```

> ### `np.identity(n, dtype=float)`
> **Definition:** Shorthand for a square identity matrix — equivalent to `np.eye(n)` but only accepts a single size, no offset.
> ```python
> np.identity(4)
> ```

> ### `np.arange([start,] stop[, step], dtype=None)`
> **Definition:** The array equivalent of Python's `range()` — generates evenly spaced values within a half-open interval `[start, stop)`.
> **Parameters:** `start` — first value (default 0). `stop` — exclusive upper bound. `step` — spacing between values (default 1). Works with floats too, but float step sizes can produce a slightly unexpected number of elements due to rounding.
> ```python
> np.arange(5)          # [0 1 2 3 4]
> np.arange(2, 10, 2)   # [2 4 6 8]
> ```

> ### `np.linspace(start, stop, num=50, endpoint=True)`
> **Definition:** Generates `num` evenly spaced values between `start` and `stop`, inclusive of the endpoint by default.
> **Parameters:** `start`, `stop` — the range boundaries. `num` — how many points to generate. `endpoint` — whether to include `stop` itself.
> **Returns:** array of `num` values.
> ```python
> np.linspace(0, 1, 5)     # [0.   0.25 0.5  0.75 1.  ]
> ```
> **Note:** prefer `linspace` over `arange` whenever you care about the *exact count* of points, since `arange` with float steps can overshoot or undershoot by one element due to floating-point rounding.

> ### `np.logspace(start, stop, num=50, base=10.0)`
> **Definition:** Like `linspace`, but the values are evenly spaced on a logarithmic scale — generates `base**y` for `y` linearly spaced between `start` and `stop`.
> ```python
> np.logspace(0, 2, 3)    # [1. 10. 100.]  — because 10^0, 10^1, 10^2
> ```

> ### `np.empty(shape, dtype=float)`
> **Definition:** Allocates an array of the given shape *without* initializing its values — whatever bytes happen to be in memory show up as garbage numbers.
> **Returns:** uninitialized array.
> ```python
> np.empty((2, 2))   # fast, but contents are unpredictable — always overwrite before reading
> ```
> **Note:** slightly faster than `zeros`/`ones` because it skips the fill step. Only use it when you're about to overwrite every element anyway.

> ### `np.zeros_like(a)`, `np.ones_like(a)`, `np.full_like(a, value)`, `np.empty_like(a)`
> **Definition:** Same as the base functions above, but automatically match the shape and dtype of an existing array `a` instead of you specifying them.
> ```python
> a = np.array([[1, 2], [3, 4]])
> np.zeros_like(a)     # 2x2 zeros, same dtype as a
> ```

## 3. Array attributes (not functions — properties read off an existing array)

> ### `.shape`
> **Definition:** A tuple giving the size of the array along each dimension.
> ```python
> np.zeros((2, 3)).shape   # (2, 3)
> ```

> ### `.ndim`
> **Definition:** The number of dimensions (axes) the array has.
> ```python
> np.zeros((2, 3)).ndim    # 2
> ```

> ### `.size`
> **Definition:** The total number of elements — the product of all entries in `.shape`.
> ```python
> np.zeros((2, 3)).size    # 6
> ```

> ### `.dtype`
> **Definition:** The data type of every element in the array (all elements share one type).
> ```python
> np.array([1, 2, 3]).dtype   # dtype('int64')
> ```

> ### `.itemsize`
> **Definition:** The number of bytes each single element occupies.
> ```python
> np.array([1], dtype=np.int32).itemsize   # 4
> ```

> ### `.nbytes`
> **Definition:** Total memory the array's data consumes: `size * itemsize`.
> ```python
> np.zeros((100, 100), dtype=np.float64).nbytes   # 80000
> ```

## 4. Data types (`dtype`)

Common dtypes: `int8/16/32/64`, `uint8/16/32/64` (unsigned), `float16/32/64`, `bool`, `complex64/128`, `<U10` (fixed-width 10-character Unicode string).

> ### `.astype(dtype, copy=True)`
> **Definition:** Converts an array to a new dtype, returning a new array (never modifies in place).
> **Parameters:** `dtype` — target type. `copy` — if `False` and the dtype already matches, may skip copying.
> ```python
> a = np.array([1.7, 2.9])
> a.astype(np.int32)    # [1 2] — truncates, does not round
> ```
> **Note:** converting float → int truncates toward zero, it does not round. Use `np.round()` first if you want rounding.

> **Overflow behavior:** narrow integer dtypes wrap around silently instead of raising an error.
> ```python
> np.array([200], dtype=np.int8) + 100   # [44] — silently overflowed; int8 maxes out at 127
> ```

## 5. Indexing and slicing (core syntax, not functions)

> ### Basic slicing: `a[start:stop:step]`
> **Definition:** Selects a range of elements along an axis, same semantics as Python list slicing but extended to N dimensions with comma-separated axes.
> ```python
> a = np.arange(10)
> a[2:5]      # [2 3 4]
> a[::2]      # every other element: [0 2 4 6 8]
> a[::-1]     # reversed
>
> m = np.arange(12).reshape(3, 4)
> m[1, 2]       # single element at row 1, col 2
> m[:, 1]       # entire column 1
> m[0:2, 1:3]   # sub-block: rows 0-1, cols 1-2
> ```
> **Note:** basic slices return **views** — they share memory with the original array. Modifying a slice modifies the source array too. See Part 3, Section 17.

---

# part2 - intermediate


## 6. Broadcasting

**Definition:** the set of rules NumPy uses to perform element-wise operations on arrays of different shapes without copying data. Comparing shapes from the right: dimensions are compatible if they're equal, or if one of them is `1` (which gets stretched to match).

**Rules of broadcasting:**
1. Compare array shapes from the **trailing dimensions** moving left.
2. Two dimensions are compatible if they are **equal** or if one of them is **1**.
3. If one array has fewer dimensions, NumPy treats it as if it had leading `1`s.
4. The resulting shape uses the **largest size** in each compatible dimension.
5. If any dimension pair is incompatible, NumPy raises a `ValueError`.

{
    1. if the dimension is not same 
    add padding(1) to left side.
    2.
}

```python
x = np.array([1, 2, 3])           # shape (3,)
y = np.array([[1], [2], [3]])     # shape (3, 1)
x + y
# broadcasts to (3, 3):
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]]
```

> ### `.mean(axis=None, keepdims=False)` and `keepdims`
> **Definition:** `keepdims=True` keeps a reduced axis in the output as size 1 instead of dropping it, which keeps the result broadcastable against the original array.
> ```python
> data = np.array([[1, 2, 3], [4, 5, 6]])
> row_means = data.mean(axis=1, keepdims=True)   # shape (2, 1), not (2,)
> centered = data - row_means                      # broadcasts cleanly
> ```

## 7. Reshaping functions

> ### `.reshape(shape)`
> **Definition:** Returns an array with the same data but a different shape. Total element count must stay the same.
> **Parameters:** `shape` — new shape as a tuple; one dimension may be `-1`, meaning "infer this size automatically."
> ```python
> a = np.arange(12)
> a.reshape(3, 4)     # 3 rows, 4 cols
> a.reshape(3, -1)     # -1 inferred as 4
> ```
> **Returns:** a view when the data layout allows it, otherwise a copy.

> ### `.flatten()`
> **Definition:** Collapses an array to 1-D. **Always returns a copy.**
> ```python
> np.array([[1, 2], [3, 4]]).flatten()   # [1 2 3 4]
> ```

> ### `.ravel()`
> **Definition:** Collapses an array to 1-D, same result as `.flatten()`, but returns a **view** when possible (faster, no copy) and only falls back to a copy when the data isn't contiguous.
> ```python
> np.array([[1, 2], [3, 4]]).ravel()
> ```

> ### `.T` / `.transpose(axes=None)`
> **Definition:** Reverses (or permutes) the array's axes. For a 2-D array, this is the standard matrix transpose. Returns a **view**, not a copy.
> ```python
> np.arange(6).reshape(2, 3).T.shape   # (3, 2)
> ```
> **Parameters (for `.transpose()`):** `axes` — optional tuple specifying a custom axis order for arrays with more than 2 dimensions.

## 8. Joining and splitting functions

> ### `np.concatenate(arrays, axis=0)`
> **Definition:** Joins a sequence of arrays along an existing axis. Arrays must match in every dimension except the join axis.
> ```python
> np.concatenate([np.array([1,2]), np.array([3,4])])   # [1 2 3 4]
> ```

> ### `np.stack(arrays, axis=0)`
> **Definition:** Joins arrays along a **brand-new** axis (unlike `concatenate`, which uses an existing one). All input arrays must have identical shape.
> ```python
> np.stack([np.array([1,2,3]), np.array([4,5,6])])   # shape (2, 3)
> ```

> ### `np.vstack(arrays)` / `np.hstack(arrays)`
> **Definition:** Convenience wrappers around `concatenate`/`stack` for the common cases: `vstack` stacks arrays row-wise (vertically), `hstack` stacks them column-wise (horizontally / end-to-end for 1-D arrays).
> ```python
> np.vstack([np.array([1,2,3]), np.array([4,5,6])])   # shape (2, 3)
> np.hstack([np.array([1,2,3]), np.array([4,5,6])])   # shape (6,)
> ```

> ### `np.split(array, indices_or_sections, axis=0)`
> **Definition:** Splits an array into multiple sub-arrays along an axis. If given an integer `N`, splits into `N` equal pieces (raises an error if it doesn't divide evenly — use `array_split` if it might not).
> ```python
> np.split(np.arange(12).reshape(3,4), 3)   # 3 pieces along axis 0
> ```

> ### `np.array_split(array, sections, axis=0)`
> **Definition:** Like `np.split`, but tolerates section counts that don't divide the array evenly, distributing the remainder across the first few pieces.

> ### `np.hsplit(array, sections)` / `np.vsplit(array, sections)`
> **Definition:** Convenience shortcuts for splitting along columns (`hsplit`) or rows (`vsplit`).

## 9. Boolean and fancy indexing

> ### Boolean indexing: `a[condition]`
> **Definition:** Selects elements where a boolean array (usually produced by a comparison) is `True`. Always returns a **copy**.
> ```python
> a = np.array([1, -2, 3, -4, 5])
> a[a > 0]          # [1 3 5]
> a[a < 0] = 0       # conditional in-place assignment: zeroes out negatives
> ```

> ### `np.where(condition, x, y)`
> **Definition:** The vectorized equivalent of an if/else — for each position, returns `x` if `condition` is True there, else `y`. If called with only `condition`, returns the indices where it's True instead.
> **Parameters:** `condition` — boolean array. `x`, `y` — values or arrays to choose from (broadcastable to condition's shape).
> ```python
> a = np.array([1, -2, 3, -4])
> np.where(a > 0, a, 0)     # [1 0 3 0]
> np.where(a > 0)            # (array([0, 2]),) — indices, as a tuple of arrays
> ```


> ### `np.select(condlist, choicelist, default=0)`
> **Definition:** A multi-branch version of `np.where` — evaluates a list of conditions in order and picks the corresponding choice for the first one that's True at each position.
> ```python
> a = np.array([1, -1, 2])
> np.select([a < 0, a > 0], ['neg', 'pos'], default='zero')   # ['pos' 'neg' 'pos']
> ```

> ### Fancy indexing: `a[index_array]`
> **Definition:** Indexing with an array (or list) of integer indices instead of a slice. Always returns a **copy**.
> ```python
> a = np.array([10, 20, 30, 40, 50])
> a[[0, 2, 4]]     # [10 30 50]
>
> m = np.arange(12).reshape(3, 4)
> m[[0, 2], [1, 3]]   # element-paired: picks (0,1) and (2,3) -> [1 11], NOT a grid
> ```

## 10. Universal functions (ufuncs)

**Definition:** functions that operate element-wise on arrays, implemented as fast compiled loops. This is the mechanism behind almost every fast NumPy operation.

> ### `np.sqrt(x)`, `np.exp(x)`, `np.log(x)`, `np.log2(x)`, `np.log10(x)`
> **Definition:** Element-wise square root, exponential (e^x), natural log, base-2 log, base-10 log.
> ```python
> np.sqrt(np.array([1, 4, 9]))   # [1. 2. 3.]
> ```

> ### `np.sin(x)`, `np.cos(x)`, `np.tan(x)`
> **Definition:** Element-wise trigonometric functions, input in radians.

> ### `np.abs(x)`
> **Definition:** Element-wise absolute value.

> ### `np.add(x, y)`, `np.subtract`, `np.multiply`, `np.divide`, `np.power`, `np.mod`
> **Definition:** Element-wise arithmetic — the underlying functions behind the `+ - * / ** %` operators. Calling the function directly lets you pass an `out=` array to write results into (avoiding a new allocation).
> ```python
> result = np.empty(3)
> np.add(np.array([1,2,3]), np.array([4,5,6]), out=result)
> ```

> ### `np.maximum(x, y)` / `np.minimum(x, y)`
> **Definition:** Element-wise max/min between two arrays (broadcastable). Different from `np.max`/`np.min`, which reduce a single array to a summary value.
> ```python
> np.maximum(np.array([1, 5, 3]), 4)   # [4 5 4]
> ```

> ### `ufunc.reduce(array)`
> **Definition:** Repeatedly applies a ufunc across an array to collapse it to a single value — e.g. `np.add.reduce` is equivalent to `np.sum`.
> ```python
> np.add.reduce([1, 2, 3, 4])   # 10
> ```

> ### `ufunc.accumulate(array)`
> **Definition:** Like `.reduce`, but keeps every intermediate result instead of only the final one — a running total.
> ```python
> np.multiply.accumulate([1, 2, 3, 4])   # [1 2 6 24]
> ```

> ### `ufunc.outer(a, b)`
> **Definition:** Applies the ufunc to every pair `(a[i], b[j])`, producing a full outer-product-style grid.
> ```python
> np.add.outer(np.array([1,2,3]), np.array([10,20]))
> # [[11 21]
> #  [12 22]
> #  [13 23]]
> ```

## 11. Aggregation functions and the `axis` argument

> ### `.sum(axis=None)`, `.mean(axis=None)`, `.std(axis=None)`, `.var(axis=None)`, `.min(axis=None)`, `.max(axis=None)`, `.prod(axis=None)`
> **Definition:** Reduce an array to a summary statistic, either over the whole array (`axis=None`, the default) or along one specific axis.
> **Parameters:** `axis` — which dimension to collapse. **Mental model: `axis=N` means "collapse dimension N"** — the result loses that dimension (unless `keepdims=True`).
> ```python
> m = np.array([[1, 2, 3], [4, 5, 6]])
> m.sum()             # 21 — everything
> m.sum(axis=0)       # [5 7 9]  — collapses rows, sums down each column
> m.sum(axis=1)       # [6 15]    — collapses columns, sums across each row
> ```

> ### `.argmin(axis=None)` / `.argmax(axis=None)`
> **Definition:** Returns the *index* of the minimum/maximum element, rather than its value. With no axis, indexes into the flattened array.
> ```python
> np.array([3, 1, 4]).argmax()   # 2
> ```

> ### `.cumsum(axis=None)` / `.cumprod(axis=None)`
> **Definition:** Cumulative sum / cumulative product — like `.sum()`/`.prod()` but keeps every running intermediate value instead of collapsing to one number.
> ```python
> np.array([1, 2, 3, 4]).cumsum()    # [1 3 6 10]
> ```

> ### `.all(axis=None)` / `.any(axis=None)`
> **Definition:** `.all()` returns `True` if every element is truthy (non-zero); `.any()` returns `True` if at least one element is. Commonly used with a comparison to check a condition across an array.
> ```python
> (np.array([1, 2, 3]) > 0).all()   # True
> ```

## 12. Sorting and searching functions

> ### `np.sort(a, axis=-1)`
> **Definition:** Returns a **sorted copy** of the array. `a.sort()` (method form) sorts **in place** instead and returns `None`.
> **Parameters:** `axis` — axis to sort along (default: the last axis)....

> ### `np.argsort(a, axis=-1)`
> **Definition:** Returns the *indices* that would sort the array, rather than the sorted values themselves. Useful for sorting one array based on the order of another.
> ```python
> np.argsort(np.array([3, 1, 2]))   # [1 2 0]
> ```

> ### `np.searchsorted(sorted_array, values)`
> **Definition:** For each value, finds the index where it should be inserted into `sorted_array` to keep it sorted (binary search — assumes the input is already sorted).
> ```python
> np.searchsorted(np.array([1, 3, 5, 7]), [4, 0, 8])   # [2 0 4]
> ```

> ### `np.unique(a, return_counts=False, return_index=False)`
> **Definition:** Returns the sorted unique elements of an array.
> **Parameters:** `return_counts` — also return how many times each unique value appears. `return_index` — also return the index of the first occurrence of each.
> ```python
> np.unique(np.array([3, 1, 2, 1, 3]), return_counts=True)
> # (array([1, 2, 3]), array([2, 1, 2]))
> ```

> ### `np.partition(a, kth)`
> **Definition:** Rearranges elements so that everything before index `kth` is ≤ the element at `kth`, and everything after is ≥ it — a faster alternative to a full sort when you only need e.g. "the k smallest values" without caring about their exact order.
> ```python
> np.partition(np.array([3,1,4,1,5,9,2,6]), 3)   # [1 1 2 3 4 5 6 9] (may vary in unsorted regions)
> ```

> ### `np.argpartition(a, kth)`
> **Definition:** Same as `np.partition`, but returns indices instead of values — the index-version counterpart, same relationship as `argsort` is to `sort`.

## 13. Random number generation

The modern interface is the `Generator` class via `np.random.default_rng()` — prefer this over the legacy `np.random.seed()`/`np.random.rand()` functions in new code.

> ### `np.random.default_rng(seed=None)`
> **Definition:** Creates a `Generator` object — a self-contained random number source. Passing a `seed` makes the sequence of "random" values reproducible.
> **Returns:** a `Generator` instance with methods below.
> ```python
> rng = np.random.default_rng(seed=42)
> ```

> ### `rng.random(size=None)`
> **Definition:** Draws floats uniformly from `[0, 1)`.
> ```python
> rng.random(5)   # 5 floats in [0, 1)
> ```

> ### `rng.integers(low, high=None, size=None)`
> **Definition:** Draws random integers from `[low, high)` (exclusive of `high`).
> ```python
> rng.integers(0, 10, size=5)
> ```

> ### `rng.normal(loc=0.0, scale=1.0, size=None)`
> **Definition:** Draws from a normal (Gaussian) distribution. `loc` is the mean, `scale` is the standard deviation.
> ```python
> rng.normal(loc=0, scale=1, size=(3, 3))
> ```

> ### `rng.uniform(low=0.0, high=1.0, size=None)`
> **Definition:** Draws floats uniformly from `[low, high)` — a generalized version of `.random()` with custom bounds.

> ### `rng.choice(a, size=None, replace=True)`
> **Definition:** Randomly samples elements from array `a`.
> **Parameters:** `replace` — if `False`, each element can be picked at most once (sampling without replacement).
> ```python
> rng.choice([1, 2, 3, 4, 5], size=3, replace=False)
> ```

> ### `rng.shuffle(a)`
> **Definition:** Shuffles an array **in place** along its first axis. Returns `None`.

> ### `rng.permutation(x)`
> **Definition:** Like `.shuffle()`, but returns a **new shuffled copy** instead of modifying in place. If `x` is an integer, shuffles `np.arange(x)`.
> ```python
> rng.permutation(5)   # a random ordering of [0, 1, 2, 3, 4]
> ```

## 14. Linear algebra basics

> ### `@` operator / `np.dot(a, b)` / `np.matmul(a, b)`
> **Definition:** Matrix multiplication. `@` is the preferred modern syntax. `np.dot` and `np.matmul` behave identically for 2-D arrays but diverge slightly for higher dimensions (`matmul` treats stacks of matrices batch-wise; `dot` does a more general tensor contraction).
> ```python
> A = np.array([[1, 2], [3, 4]])
> B = np.array([[5, 6], [7, 8]])
> A @ B
> ```
> **Note:** `A * B` is element-wise multiplication, *not* matrix multiplication — a very common source of bugs.

> ### `np.linalg.inv(a)`
> **Definition:** Computes the matrix inverse. Raises `LinAlgError` if the matrix is singular (non-invertible).

> ### `np.linalg.det(a)`
> **Definition:** Computes the determinant of a square matrix.

> ### `np.trace(a)`
> **Definition:** Sum of the elements along the main diagonal.

---

# Part 3 — Advanced

## 15. Views, copies, and memory layout

> ### `.strides`
> **Definition:** A tuple giving the number of bytes to step forward in memory to move to the next element along each axis. This is the mechanism underneath why reshaping/transposing are (almost) free.
> ```python
> a = np.arange(24).reshape(4, 6)
> a.strides    # (48, 8) — 48 bytes to next row, 8 bytes to next column (int64 = 8 bytes)
> ```

> ### `np.shares_memory(a, b)`
> **Definition:** Reliably checks whether two arrays' data buffers overlap — i.e. whether modifying one could affect the other. **This is the correct way to check for a view relationship** — checking `b.base is a` can give a false negative, because views can chain through multiple layers back to a shared root buffer.
> ```python
> a = np.arange(6)
> b = a.reshape(2, 3)
> np.shares_memory(a, b)   # True
> ```

> **Which operations return views vs. copies:**

| Operation | Result |
|---|---|
| Basic slicing (`a[1:3]`, `a[:, 0]`) | View |
| `.reshape()` (contiguous data) | View |
| `.T` / `.transpose()` | View |
| `.ravel()` | View when possible, else copy |
| Boolean indexing (`a[a > 0]`) | Copy |
| Fancy indexing (`a[[0, 2, 4]]`) | Copy |
| `.flatten()` | Always copy |
| Arithmetic (`a + b`) | Always a new array |
| `.copy()` | Explicit, forced copy |

> ### `.copy()`
> **Definition:** Forces an independent copy of the array's data, guaranteeing no memory sharing with the original, regardless of what operation produced it.

> ### `order='C'` vs `order='F'`
> **Definition:** Controls memory layout. `'C'` (row-major, the default) stores each row contiguously; `'F'` (column-major, Fortran-style) stores each column contiguously.
> ```python
> a = np.arange(6).reshape(2, 3, order='C')
> b = np.arange(6).reshape(2, 3, order='F')
> ```
> **Note:** iterating along the contiguous axis is cache-friendly and faster; iterating the other way causes scattered memory access on large arrays.

## 16. Vectorization and performance

**Core principle:** avoid Python-level `for` loops over individual array elements. Every ufunc/operation you use instead runs as a compiled loop.

```python
# Slow: Python loop
result = np.empty_like(a)
for i in range(len(a)):
    result[i] = a[i] ** 2 + 1

# Fast: vectorized
result = a ** 2 + 1
```

> ### `out=` parameter
> **Definition:** Most ufuncs accept an `out=` array to write results directly into, avoiding a new memory allocation on every call — useful in loops that repeat the same operation many times.
> ```python
> result = np.empty(1000)
> np.add(a, b, out=result)
> ```

> ### In-place operators (`+=`, `*=`, etc.)
> **Definition:** Modify an array's existing buffer directly instead of allocating a new array. `a += 1` reuses `a`'s memory; `a = a + 1` allocates a fresh array and rebinds the name.

## 17. Structured arrays

> ### Structured `dtype`: `np.array(data, dtype=[(name, type), ...])`
> **Definition:** Defines an array where each "row" is a record with multiple named, possibly differently-typed fields — a lightweight, dependency-free alternative to a pandas DataFrame.
> ```python
> people = np.array(
>     [("Alice", 30, 55.5), ("Bob", 25, 70.2)],
>     dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")]
> )
> people["name"]           # array(['Alice', 'Bob'])
> people["age"].mean()      # 27.5
> people[people["age"] > 26]   # boolean indexing filters whole records
> ```

## 18. Masked arrays

> ### `numpy.ma.masked_equal(a, value)`
> **Definition:** Creates a masked array where every element equal to `value` is flagged as invalid and excluded from subsequent calculations (mean, sum, etc.), without deleting the data.
> ```python
> import numpy.ma as ma
> data = np.array([1, 2, -999, 4, 5])
> masked = ma.masked_equal(data, -999)
> masked.mean()      # 3.0 — the -999 sentinel is excluded
> ```

> ### `.filled(fill_value)`
> **Definition:** Returns a regular (non-masked) array with masked entries replaced by `fill_value`.
> ```python
> masked.filled(0)   # [1 2 0 4 5]
> ```

## 19. `np.einsum(subscripts, *operands)`

**Definition:** Einstein summation notation — one expressive syntax covering matrix multiplication, transposition, trace, dot products, and general tensor contractions, often fused into a single fast pass instead of chaining multiple separate NumPy calls.

**Parameters:** `subscripts` — a string like `'ij,jk->ik'`. Letters label axes of each input; repeated letters across inputs mean "multiply and sum over that axis"; the part after `->` lists which axes survive into the output.

```python
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
np.einsum('ij,jk->ik', A, B)     # same as A @ B
np.einsum('ii', np.eye(3))         # trace: 3.0
np.einsum('ij->ji', A)              # transpose
np.einsum('ij,ij->i', A, A)        # row-wise sum of squares
```

## 20. Advanced linear algebra

> ### `np.linalg.eig(a)`
> **Definition:** Computes eigenvalues and eigenvectors of a square matrix.
> **Returns:** a tuple `(eigenvalues, eigenvectors)` — eigenvectors are the columns of the returned matrix.
> ```python
> vals, vecs = np.linalg.eig(np.array([[4, 2], [1, 3]]))
> ```

> ### `np.linalg.svd(a)`
> **Definition:** Singular Value Decomposition — factorizes a matrix as `U @ diag(S) @ Vt`. Core to PCA, dimensionality reduction, and low-rank approximation.
> **Returns:** tuple `(U, S, Vt)`.
> ```python
> U, S, Vt = np.linalg.svd(np.array([[1, 2], [3, 4]]))
> ```

> ### `np.linalg.solve(a, b)`
> **Definition:** Solves the linear system `a @ x = b` for `x` directly. Prefer this over `np.linalg.inv(a) @ b` — it's both faster and numerically more stable.
> ```python
> x = np.linalg.solve(np.array([[3, 1], [1, 2]]), np.array([9, 8]))
> ```

> ### `np.linalg.norm(a, ord=None)`
> **Definition:** Computes a vector or matrix norm — a measure of "size."
> **Parameters:** `ord` — which norm to use (default is the Euclidean/L2 norm; `ord=1` gives the Manhattan/L1 norm).
> ```python
> np.linalg.norm(np.array([3, 4]))          # 5.0
> np.linalg.norm(np.array([3, 4]), ord=1)   # 7.0
> ```

> ### `np.linalg.qr(a)`
> **Definition:** QR decomposition — factorizes a matrix into an orthogonal matrix `Q` and an upper-triangular matrix `R`, such that `a = Q @ R`. Used for solving least-squares problems and as a numerically stable building block in many algorithms.
> **Returns:** tuple `(Q, R)`.

> ### `np.linalg.cholesky(a)`
> **Definition:** Cholesky decomposition — factorizes a symmetric, positive-definite matrix into `L @ L.T`, where `L` is lower-triangular. Faster than a general decomposition when your matrix meets those conditions (common in covariance matrices, optimization).

> ### `np.linalg.matrix_rank(a)`
> **Definition:** Returns the numerical rank of a matrix — the number of linearly independent rows/columns.

## 21. Custom ufuncs

> ### `np.vectorize(func)`
> **Definition:** Wraps an ordinary scalar Python function so it broadcasts over arrays like a ufunc.
> ```python
> def piecewise(x):
>     return x**2 + 1 if x > 0 else 0
> vec_fn = np.vectorize(piecewise)
> vec_fn(np.array([-2, -1, 0, 1, 2]))   # [0 0 0 2 5]
> ```
> **Note:** this is a convenience wrapper, *not* a performance tool — it still calls your Python function once per element internally, so it isn't faster than a plain loop. For real speed on custom per-element logic, use Numba's `@njit`/`@vectorize` decorators, which actually compile the function.

## 22. Memory-mapped arrays

> ### `np.memmap(filename, dtype, mode, shape)`
> **Definition:** Treats a file on disk as if it were an in-memory `ndarray`, loading only the portions you actually access — for datasets too large to fit in RAM.
> **Parameters:** `filename` — path to the file. `dtype` — element type. `mode` — `'r'` (read-only), `'r+'` (read/write, file must exist), `'w+'` (create/overwrite). `shape` — array shape.
> ```python
> arr = np.memmap('data.dat', dtype='float32', mode='w+', shape=(1000, 1000))
> arr[:] = np.random.rand(1000, 1000)
> arr.flush()     # ensure writes hit disk
> ```

## 23. Useful utility functions

> ### `np.clip(a, a_min, a_max)`
> **Definition:** Limits every element to a given range — values below `a_min` become `a_min`, values above `a_max` become `a_max`.
> ```python
> np.clip(np.array([-5, 0, 5, 10]), 0, 5)   # [0 0 5 5]
> ```

> ### `np.tile(a, reps)`
> **Definition:** Repeats the entire array `reps` times, tiling it like floor tiles.
> ```python
> np.tile(np.array([1, 2]), 3)   # [1 2 1 2 1 2]
> ```

> ### `np.repeat(a, repeats)`
> **Definition:** Repeats each individual element `repeats` times (different from `tile`, which repeats the whole array).
> ```python
> np.repeat(np.array([1, 2]), 3)   # [1 1 1 2 2 2]
> ```

> ### `np.pad(a, pad_width, constant_values=0)`
> **Definition:** Adds padding around an array.
> **Parameters:** `pad_width` — how many elements to add before/after each axis. `constant_values` — the fill value for the padding.
> ```python
> np.pad(np.array([1, 2, 3]), (1, 2), constant_values=0)   # [0 1 2 3 0 0]
> ```

> ### `np.meshgrid(x, y)`
> **Definition:** Builds coordinate grids from two 1-D arrays — turns a list of x-values and a list of y-values into full 2-D grids of every (x, y) combination, useful for evaluating functions over a 2-D surface or plotting.
> ```python
> xx, yy = np.meshgrid(np.array([1, 2]), np.array([3, 4, 5]))   # both shape (3, 2)
> ```

> ### `np.apply_along_axis(func, axis, a)`
> **Definition:** Applies a Python function to 1-D slices of an array along the given axis. A fallback for logic that can't be vectorized directly — slower than a true ufunc, but more flexible.
> ```python
> np.apply_along_axis(np.sum, 1, np.arange(6).reshape(2, 3))   # [3 12]
> ```

> ### `np.isnan(a)` / `np.nan_to_num(a)`
> **Definition:** `isnan` returns a boolean array flagging `NaN` positions. `nan_to_num` replaces `NaN` with 0 (and infinities with large finite numbers) by default.
> ```python
> np.isnan(np.array([1, np.nan]))     # [False True]
> np.nan_to_num(np.array([1, np.nan, np.inf]))
> ```

> ### `np.nanmean(a)`, `np.nansum(a)`, `np.nanmax(a)`, etc.
> **Definition:** NaN-aware versions of the standard aggregations — ignore `NaN` values instead of letting them propagate and turn the whole result into `NaN`.
> ```python
> np.nanmean(np.array([1, np.nan, 3]))   # 2.0
> ```

> ### `np.set_printoptions(precision=None, suppress=None, ...)`
> **Definition:** Controls how arrays are displayed when printed — decimal precision, whether to suppress scientific notation, line width, etc. Affects display only, never the underlying data.
> ```python
> np.set_printoptions(precision=2, suppress=True)
> ```

---

## Where to go next

- **pandas** — built on NumPy, adds labeled rows/columns and heterogeneous data handling.
- **Numba** — JIT-compile your own Python loops to machine-code speed with a decorator.
- **SciPy** — adds optimization, statistics, signal processing, sparse matrices on top of NumPy.
- **CuPy** — a near drop-in NumPy replacement that runs on GPU.
- **PyTorch/JAX arrays** — if you're heading toward machine learning, their array APIs are deliberately close to NumPy's, so everything here transfers directly.

A good way to consolidate this: pick a small dataset and re-implement something you'd normally do with loops (a moving average, a distance matrix between points, an image filter) entirely with vectorized NumPy — no `for` loops over elements at all.
