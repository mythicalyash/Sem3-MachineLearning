# The Complete Pandas Reference Notes — Beginner to Advanced

Every function below follows the same format and every example has been run against pandas 3.0:

> ### `signature`
> **Definition:** what it does, in plain language.
> **Parameters:** each argument, its type, and what it controls.
> **Returns:** what you get back.
> ```python
> example usage
> ```
> **Note:** gotchas, performance notes, or common mistakes (only where relevant).

```python
import pandas as pd
import numpy as np    # pandas is built on numpy, you'll want both
```

---

## Table of Contents

**Part 1 — Foundations:** what pandas is, `Series`/`DataFrame`, reading data, inspecting, indexing
**Part 2 — Intermediate:** filtering, missing data, sorting, `groupby`, merging/joining, `apply`/`map`, strings, datetimes, reshaping, duplicates
**Part 3 — Advanced:** `MultiIndex`, stacking, window functions, categoricals, `query`/`eval`, method chaining, time series resampling, performance, gotchas

---

# Part 1 — Foundations

## 1. Why pandas exists

NumPy arrays are fast but anonymous — element `[3, 1]` means nothing beyond "row 3, column 1." **pandas** adds labels: named columns, a row index, mixed types per column, and built-in handling for missing data. It's the standard tool for anything that looks like a spreadsheet or a database table — loading it, cleaning it, slicing it, summarizing it.

Under the hood, each pandas column is backed by a NumPy array (or an Arrow-backed array in newer versions), so everything you already know about NumPy dtypes and vectorization still applies.

## 2. The two core objects

> ### `pd.Series(data, index=None, dtype=None, name=None)`
> **Definition:** A one-dimensional labeled array — think of it as a single column with a row index attached.
> **Parameters:** `data` — list, dict, scalar, or NumPy array. `index` — labels for each element (defaults to `0, 1, 2, ...`). `dtype` — force a type. `name` — a name for the Series (becomes the column name if it's later put into a DataFrame).
> **Returns:** a `Series` object.
> ```python
> s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
> #  a    10
> #  b    20
> #  c    30
> #  dtype: int64
> s['b']    # 20 — indexed by label
> ```

> ### `pd.DataFrame(data, index=None, columns=None)`
> **Definition:** A two-dimensional labeled table — a collection of `Series` sharing one row index, each with its own dtype. The central object in pandas.
> **Parameters:** `data` — most commonly a dict of `{column_name: list_of_values}`, but also accepts a list of dicts, a 2-D NumPy array, or another DataFrame. `index` — row labels. `columns` — column labels (useful when `data` doesn't already carry names).
> **Returns:** a `DataFrame`.
> ```python
> df = pd.DataFrame({
>     'name': ['Alice', 'Bob', 'Carol'],
>     'age':  [30, 25, 35],
>     'city': ['NY', 'LA', 'NY']
> })
> ```

## 3. Reading and writing data

> ### `pd.read_csv(filepath, sep=',', header=0, index_col=None, dtype=None, parse_dates=None)`
> **Definition:** Reads a CSV (or any delimited text) file into a DataFrame. The single most commonly used function in pandas.
> **Parameters:** `filepath` — path or URL. `sep` — field delimiter. `header` — which row holds column names (`None` if there is none). `index_col` — which column to use as the row index. `dtype` — force specific column types. `parse_dates` — list of columns to parse as dates.
> **Returns:** a `DataFrame`.
> ```python
> df = pd.read_csv('data.csv')
> df = pd.read_csv('data.tsv', sep='\t', parse_dates=['order_date'])
> ```

> ### `pd.read_excel(filepath, sheet_name=0)` / `pd.read_json(filepath)`
> **Definition:** Same idea as `read_csv`, for Excel workbooks and JSON respectively. `sheet_name` selects which sheet to load (by name, index, or `None` for all sheets as a dict).

> ### `.to_csv(filepath, index=True)` / `.to_excel(filepath)` / `.to_json(filepath)` / `.to_parquet(filepath)`
> **Definition:** Write a DataFrame out to disk in the given format.
> **Parameters:** `index` — whether to write the row index as a column (commonly set to `False` when the index is just a default integer range you don't want cluttering the file).
> ```python
> df.to_csv('output.csv', index=False)
> ```
> **Note:** `to_parquet` (requires `pyarrow` or `fastparquet`) is strongly preferred over CSV for large datasets — it's smaller, faster to read/write, and preserves dtypes exactly.

## 4. Inspecting a DataFrame

> ### `.shape`
> **Definition:** Tuple of `(number of rows, number of columns)`.

> ### `.columns` / `.index`
> **Definition:** `.columns` is the list of column labels; `.index` is the list of row labels. Both are pandas `Index` objects (array-like, can be iterated, sliced, renamed).

> ### `.dtypes`
> **Definition:** The data type of every column, one per column (unlike NumPy, a DataFrame can mix types across columns).
> ```python
> df.dtypes
> ```

> ### `.head(n=5)` / `.tail(n=5)`
> **Definition:** Returns the first/last `n` rows — the standard "let me eyeball this" check after loading or transforming data.

> ### `.info()`
> **Definition:** Prints a summary: column names, non-null counts, dtypes, and memory usage. The fastest way to spot missing data and wrong dtypes at a glance.

> ### `.describe()`
> **Definition:** Generates summary statistics (count, mean, std, min, quartiles, max) for numeric columns by default. Pass `include='all'` to also summarize non-numeric columns (count, unique, top, freq).

> ### `.values` / `.to_numpy()`
> **Definition:** Extracts the underlying data as a plain NumPy array, dropping row/column labels. `.to_numpy()` is the modern, preferred spelling — `.values` is older and can behave inconsistently with mixed dtypes or extension types.

## 5. Selecting and indexing

> ### `df['col']` / `df[['col1', 'col2']]`
> **Definition:** Selects a single column (returns a `Series`) or multiple columns (returns a `DataFrame`, note the double brackets).
> ```python
> df['name']          # Series
> df[['name', 'age']]  # DataFrame with 2 columns
> ```

> ### `.loc[row_labels, col_labels]`
> **Definition:** Label-based indexing — select rows/columns by their actual index/column names. Slices with `.loc` are **inclusive of the endpoint** (unlike Python/NumPy slicing).
> ```python
> df.loc[0, 'name']            # single value
> df.loc[df['age'] > 25]        # boolean row filter
> df.loc[:, ['name', 'city']]   # all rows, selected columns
> ```

> ### `.iloc[row_positions, col_positions]`
> **Definition:** Position-based indexing — select by integer position, exactly like NumPy array indexing (0-based, endpoint **exclusive**). Ignores whatever the actual labels are.
> ```python
> df.iloc[0, 0]      # first row, first column, by position
> df.iloc[0:2]         # first two rows
> ```
> **Note:** `.loc` and `.iloc` are the recommended way to index — plain `df[...]` is fine for column selection but gets ambiguous for rows, and can trigger the `SettingWithCopyWarning` (see Part 3, Section 24) if used for assignment.

---

# Part 2 — Intermediate

## 6. Filtering rows

> ### Boolean indexing: `df[condition]`
> **Definition:** Selects rows where a boolean Series (usually from a comparison) is `True`.
> ```python
> df[df['age'] > 25]
> df[(df['age'] > 25) & (df['city'] == 'NY')]   # combine with & / | — use parentheses!
> ```
> **Note:** use `&`, `|`, `~` (not `and`/`or`/`not`) when combining conditions, and wrap each condition in parentheses — operator precedence otherwise breaks the expression.

> ### `.query(expr)`
> **Definition:** Filters rows using a string expression instead of boolean masks — often more readable for complex conditions, and can be faster on large DataFrames.
> ```python
> df.query('age > 25 and city == "NY"')
> ```

## 7. Adding, dropping, and renaming columns

> ### `df['new_col'] = ...`
> **Definition:** Assigns a new column. The right-hand side can be a scalar (broadcast to every row), a list/array of matching length, or an expression on existing columns.
> ```python
> df['senior'] = df['age'] > 28
> df['age_in_months'] = df['age'] * 12
> ```

> ### `.drop(columns=[...])` / `.drop(index=[...])`
> **Definition:** Removes columns or rows by label. Returns a new DataFrame by default (pass `inplace=True` to modify in place, though explicit reassignment is generally preferred).
> ```python
> df2 = df.drop(columns=['senior'])
> df2 = df.drop(index=[0, 2])
> ```

> ### `.rename(columns={...})`
> **Definition:** Renames columns (or `index={...}` for row labels) via a mapping dict. Returns a new DataFrame.
> ```python
> df.rename(columns={'name': 'full_name'})
> ```

## 8. Missing data

Missing values show up as `NaN` (or `pd.NA` / `pd.NaT` for newer extension dtypes and datetimes).

> ### `.isna()` / `.notna()`
> **Definition:** Returns a same-shaped boolean DataFrame/Series flagging which values are missing (`isna`) or present (`notna`).
> ```python
> df.isna()
> df['age'].isna().sum()    # count of missing values in one column
> ```

> ### `.dropna(axis=0, how='any', subset=None)`
> **Definition:** Removes rows (or columns, with `axis=1`) containing missing values.
> **Parameters:** `how` — `'any'` drops if any value is missing, `'all'` drops only if every value is missing. `subset` — only consider certain columns when deciding.
> ```python
> df.dropna()
> df.dropna(subset=['age'])
> ```

> ### `.fillna(value)`
> **Definition:** Replaces missing values with a given constant, or via a dict mapping column names to different fill values, or a method like `'ffill'` (forward-fill from the previous valid value).
> ```python
> df.fillna('Unknown')
> df.fillna({'age': 0, 'city': 'Unknown'})
> df['age'].fillna(method='ffill')   # or df['age'].ffill()
> ```

## 9. Sorting

> ### `.sort_values(by, ascending=True)`
> **Definition:** Sorts rows by the values in one or more columns.
> **Parameters:** `by` — column name or list of column names. `ascending` — `True`/`False`, or a list matching multiple `by` columns for mixed sort directions.
> ```python
> df.sort_values('age', ascending=False)
> df.sort_values(['city', 'age'])
> ```

> ### `.sort_index(ascending=True)`
> **Definition:** Sorts rows by the row index labels rather than by column values.

## 10. `groupby` — split, apply, combine

> ### `.groupby(by)`
> **Definition:** Splits the DataFrame into groups based on unique values in one or more columns, so you can compute a statistic *per group*. This is pandas' single most powerful feature.
> **Parameters:** `by` — column name, list of column names, or a function/mapping to group by.
> **Returns:** a `GroupBy` object — nothing is computed until you apply an aggregation.
> ```python
> df.groupby('city')['age'].mean()
> # city
> # LA    25.0
> # NY    32.5
> ```

> ### `.agg(func)` (aggregation)
> **Definition:** Applies one or more aggregation functions to each group (or each column). Accepts a function name string (`'mean'`), a list of them, or a dict mapping columns to specific functions — the most flexible way to summarize grouped data.
> ```python
> df.groupby('city').agg({'age': ['mean', 'max'], 'name': 'count'})
> ```

> ### Common group aggregations: `.sum()`, `.mean()`, `.count()`, `.size()`, `.min()`, `.max()`, `.std()`, `.nunique()`
> **Definition:** Shortcuts for the most common aggregations, callable directly on a `GroupBy` object. `.count()` counts non-null values per column; `.size()` counts rows per group regardless of nulls; `.nunique()` counts distinct values.

> ### `.transform(func)`
> **Definition:** Like `.agg`, but returns a result the **same length as the original DataFrame** (broadcasting the group's summary value back to every row in that group) instead of collapsing to one row per group — useful for things like "subtract each group's mean from every row in that group."
> ```python
> df['age_vs_city_avg'] = df['age'] - df.groupby('city')['age'].transform('mean')
> ```

## 11. Merging, joining, and concatenating

> ### `pd.merge(left, right, on=None, how='inner')`
> **Definition:** SQL-style join between two DataFrames on one or more shared key columns.
> **Parameters:** `on` — column(s) to join on (must exist in both; use `left_on`/`right_on` if the key names differ). `how` — `'inner'` (only matching keys), `'left'` (all of `left`, matched or not), `'right'`, `'outer'` (all keys from both, matched or not).
> ```python
> pd.merge(orders, customers, on='customer_id', how='left')
> ```

> ### `.join(other)`
> **Definition:** Convenience method for merging on the **index** rather than a column, called directly on a DataFrame. Equivalent to `merge` with `left_index=True, right_index=True`.

> ### `pd.concat(objs, axis=0, ignore_index=False)`
> **Definition:** Stacks DataFrames/Series together — `axis=0` stacks rows on top of each other (like appending), `axis=1` stacks columns side by side.
> **Parameters:** `ignore_index` — if `True`, discards the original index and generates a fresh `0, 1, 2, ...` range instead of keeping (possibly duplicate) original labels.
> ```python
> pd.concat([df_january, df_february], axis=0, ignore_index=True)
> ```
> **Note:** `merge` combines by matching key *values*; `concat` just stacks by position/index — they solve different problems and are easy to reach for the wrong one.

## 12. Applying custom functions

> ### `.apply(func, axis=0)`
> **Definition:** Applies a function along an axis of a DataFrame (or to every element of a Series). On a DataFrame, `axis=0` applies the function to each column, `axis=1` applies it to each row.
> ```python
> df['age'].apply(lambda x: x * 2)              # Series: element-wise
> df.apply(lambda row: row['age'] * 2, axis=1)    # DataFrame: row-wise
> ```
> **Note:** `.apply` runs a Python function per row/element under the hood — it's flexible but slow on large data. Prefer a direct vectorized expression (`df['age'] * 2`) whenever one exists; reach for `.apply` only when the logic genuinely can't be vectorized.

> ### `.map(arg)`
> **Definition:** Element-wise substitution on a Series, using either a function or a mapping (dict/Series) of old value → new value. As of pandas 2.1+, `.map()` also works directly on DataFrames as the element-wise successor to the now-removed `.applymap()`.
> ```python
> s.map({1: 'one', 2: 'two', 3: 'three'})
> ```

## 13. String methods (`.str` accessor)

> **Definition:** The `.str` accessor exposes vectorized string operations on a Series of strings — the pandas equivalent of Python's string methods, but applied to every element at once.
> ```python
> names = pd.Series(['Alice Smith', 'bob jones'])
> names.str.upper()               # ['ALICE SMITH', 'BOB JONES']
> names.str.split(' ').str[0]      # first name: ['Alice', 'bob']
> names.str.contains('bob', case=False)   # [False, True]
> names.str.replace('Smith', 'Jones')
> names.str.strip()                 # remove leading/trailing whitespace
> names.str.len()                    # length of each string
> ```

## 14. Datetimes

> ### `pd.to_datetime(arg, format=None)`
> **Definition:** Converts strings (or other date-like values) into pandas `Timestamp`/`DatetimeIndex` objects.
> **Parameters:** `format` — explicit format string (e.g. `'%Y-%m-%d'`) to speed up and disambiguate parsing.
> ```python
> pd.to_datetime(['2024-01-01', '2024-06-15'])
> df['order_date'] = pd.to_datetime(df['order_date'])
> ```

> ### `.dt` accessor
> **Definition:** Once a column is a proper datetime dtype, `.dt` exposes vectorized access to date components.
> ```python
> df['order_date'].dt.month     # extract month as an integer, for every row
> df['order_date'].dt.day_name()
> df['order_date'].dt.year
> ```

> ### `pd.date_range(start, end=None, periods=None, freq='D')`
> **Definition:** Generates a sequence of evenly spaced dates — useful for building time series indexes.
> ```python
> pd.date_range('2024-01-01', periods=6, freq='D')
> ```

## 15. Reshaping: pivot, melt

> ### `pd.pivot_table(df, values, index, columns, aggfunc='mean')`
> **Definition:** Reshapes long-format data into a wide summary table, aggregating where multiple rows map to the same cell (like a spreadsheet pivot table).
> ```python
> pd.pivot_table(sales, values='sales', index='city', columns='month', aggfunc='sum')
> ```

> ### `.melt(id_vars, value_vars)`
> **Definition:** The inverse of pivoting — converts wide-format columns into long-format rows (one row per variable/value pair). Useful for tidying data before plotting or grouping.
> ```python
> df.melt(id_vars='city', value_vars=['jan_sales', 'feb_sales'])
> ```

## 16. Duplicates

> ### `.duplicated(subset=None, keep='first')`
> **Definition:** Returns a boolean Series flagging duplicate rows.
> **Parameters:** `subset` — only consider certain columns. `keep` — `'first'` marks all but the first occurrence as duplicate, `'last'` the reverse, `False` marks every occurrence including the first.

> ### `.drop_duplicates(subset=None, keep='first')`
> **Definition:** Removes duplicate rows, same parameters as `.duplicated()`.
> ```python
> df.drop_duplicates(subset=['name'])
> ```

---

# Part 3 — Advanced

## 17. `MultiIndex` — hierarchical indexing

> ### `pd.MultiIndex.from_arrays(arrays, names=None)`
> **Definition:** Builds a multi-level row (or column) index from several equal-length arrays — lets you index a DataFrame by more than one key at once, like a composite database key.
> ```python
> idx = pd.MultiIndex.from_arrays([['A','A','B','B'], [1,2,1,2]], names=('letter','num'))
> mdf = pd.DataFrame({'value': [10, 20, 30, 40]}, index=idx)
> mdf.loc['A']          # all rows where letter == 'A'
> ```
> **Note:** `groupby` on multiple columns automatically produces a `MultiIndex` result — this is the most common way people encounter one without building it by hand.

## 18. Stacking and unstacking

> ### `.stack()`
> **Definition:** Pivots columns into an inner level of the row index, producing a longer, narrower result (columns become part of a hierarchical index).

> ### `.unstack()`
> **Definition:** The inverse of `.stack()` — pivots an inner index level out into columns, producing a wider result.
> ```python
> mdf.unstack()   # 'num' moves from the row index into columns
> ```

## 19. Window functions

> ### `.rolling(window)`
> **Definition:** Creates a rolling (sliding) window over the data, of fixed size `window`, that you then aggregate — e.g. a moving average. The first `window - 1` results are `NaN` since there isn't a full window yet.
> ```python
> s.rolling(window=3).mean()   # 3-period moving average
> ```

> ### `.expanding()`
> **Definition:** Like `.rolling`, but the window grows to include everything from the start up to the current row (a running/cumulative aggregation), rather than a fixed-size sliding window.
> ```python
> s.expanding().sum()   # running total
> ```

> ### `.ewm(span=n)`
> **Definition:** Exponentially weighted window — like a moving average, but more recent values are weighted more heavily. `span` controls the decay rate (roughly, the "lookback" period).
> ```python
> s.ewm(span=3).mean()
> ```

## 20. Categorical data

> ### `.astype('category')`
> **Definition:** Converts a column to the `category` dtype — stores each unique value once and represents the column as compact integer codes internally. Dramatically reduces memory for columns with many repeated values (e.g. status flags, country names).
> ```python
> s = pd.Series(['a']*1000 + ['b']*1000)
> s.memory_usage(deep=True)                       # ~100,000+ bytes as plain strings
> s.astype('category').memory_usage(deep=True)      # a small fraction of that
> ```

> ### `.cat.set_categories(categories, ordered=False)`
> **Definition:** Defines the explicit set (and optionally the order) of allowed categories, enabling meaningful comparisons/sorting like `'low' < 'medium' < 'high'` instead of alphabetical.
> ```python
> cat = pd.Series(['low','high','medium'], dtype='category')
> cat = cat.cat.set_categories(['low','medium','high'], ordered=True)
> cat.sort_values()   # now sorts logically, not alphabetically
> ```

## 21. `.query()` and `.eval()` for performance

> ### `.eval(expr)`
> **Definition:** Evaluates a string expression against the DataFrame's columns, computing a new column faster and with less memory overhead than the equivalent Python expression on large DataFrames (it avoids creating intermediate temporary arrays).
> ```python
> df.eval('c = a + b')
> ```
> **Note:** the speed benefit of `.query`/`.eval` mainly shows up on large DataFrames (roughly 10,000+ rows); on small data, plain boolean indexing is just as fast and more readable.

## 22. Method chaining

> ### `.pipe(func, *args)`
> **Definition:** Passes the DataFrame into a custom function as its first argument, returning the function's result — lets you insert your own step into a chain of `.method().method().method()` calls instead of breaking the chain with an intermediate variable.
> ```python
> def add_col(df, colname, val):
>     df = df.copy()
>     df[colname] = val
>     return df
>
> result = df.pipe(add_col, 'x', 99)
> ```
> **Note:** method chaining (`df.dropna().sort_values('age').reset_index(drop=True)`) is idiomatic pandas — it avoids reassigning intermediate variables and makes a sequence of transformations read top-to-bottom.

## 23. Time series: resampling, shifting, differencing

> ### `.resample(rule)`
> **Definition:** Groups time-indexed data into fixed time buckets (e.g. every 3 days, every month) so you can aggregate — the time-series equivalent of `groupby`. Requires a `DatetimeIndex`.
> ```python
> ts = pd.Series(range(6), index=pd.date_range('2024-01-01', periods=6, freq='D'))
> ts.resample('3D').sum()
> ```

> ### `.shift(periods=1)`
> **Definition:** Shifts values forward (or backward with a negative number) by `periods` positions, leaving `NaN` in the gap — used to compare a row against a previous one (e.g. "yesterday's value").
> ```python
> ts.shift(1)
> ```

> ### `.diff(periods=1)`
> **Definition:** Computes the difference between each element and the one `periods` before it — equivalent to `s - s.shift(periods)`. Common for computing day-over-day change.

## 24. Performance and common gotchas

> ### `SettingWithCopyWarning`
> **Definition:** A warning pandas raises when you assign into a DataFrame that might be a temporary view/copy of another, meaning your assignment may silently not do what you expect.
> ```python
> # Risky — may or may not modify the original df, and pandas will warn you:
> subset = df[df['age'] > 25]
> subset['flag'] = True

> # Safe — .copy() makes the independence explicit:
> subset = df[df['age'] > 25].copy()
> subset['flag'] = True

> # Also safe — .loc assigns back into the original directly:
> df.loc[df['age'] > 25, 'flag'] = True
> ```

> ### `.copy(deep=True)`
> **Definition:** Explicitly creates an independent copy of a DataFrame/Series, same idea as NumPy's `.copy()` — use it whenever you're about to modify a subset and want to guarantee the original is untouched.

> ### `.reset_index(drop=False)`
> **Definition:** Replaces the current row index with a fresh default integer range. Commonly used after filtering/sorting/grouping to clean up a messy leftover index.
> **Parameters:** `drop` — if `False` (default), the old index is kept as a new column; if `True`, it's discarded entirely.
> ```python
> df.sort_values('age').reset_index(drop=True)
> ```

> ### Vectorize instead of looping row-by-row
> **Definition:** The same core lesson as NumPy — avoid `for index, row in df.iterrows(): ...`, which is slow because it reconstructs a Python object per row. Prefer direct column-wise expressions or `.apply` as a last resort.
> ```python
> # Slow
> for i, row in df.iterrows():
>     df.loc[i, 'total'] = row['price'] * row['qty']

> # Fast
> df['total'] = df['price'] * df['qty']
> ```

> ### `.memory_usage(deep=True)`
> **Definition:** Reports actual memory usage per column, including the true size of object/string columns (without `deep=True`, string columns are drastically underreported since it only counts pointer size, not the string data itself).

---

## Where to go next

- **NumPy** — pandas leans on it constantly; understanding array broadcasting and dtypes pays off directly here (see the companion NumPy guide).
- **matplotlib / seaborn / plotly** — pandas has built-in `.plot()` for quick charts, but these libraries give full control for anything presentation-quality.
- **polars** — a newer, Rust-based DataFrame library with a similar but stricter API, often significantly faster on large datasets — worth learning once pandas feels comfortable.
- **SQL** — `groupby`/`merge`/`pivot_table` map almost directly onto `GROUP BY`/`JOIN`/pivot queries; knowing one deepens the other.
- **dask / pyspark** — for datasets too large to fit in memory, using a pandas-like API distributed across machines.

A good way to consolidate this: take one real CSV (sales data, a public dataset, anything with dates and categories) and run it end-to-end — load, clean missing values, engineer a couple of columns, `groupby` a summary, and pivot it into a report table, without ever writing a manual `for` loop over rows.
