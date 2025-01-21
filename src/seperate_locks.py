# %%
import pandas as pd


df = pd.read_csv('data/fixed/row_locks.csv')
df.columns = df.columns.str.strip()
# Remove all trailing whitespace from TABNAME and TABSCHEMA
# NOTE: This seems to be a problem in the table lock data.
df["TABNAME"] = df["TABNAME"].astype(str).apply(lambda x: x.rstrip())
df["TABSCHEMA"] = df["TABSCHEMA"].astype(str).apply(lambda x: x.rstrip())
unique_tables = df[df['TABSCHEMA'] != 'SYSIBM']['TABNAME'].unique()


# %%
unique_tables

# %%
for table in unique_tables:
    print(f"Table: {table}")
    df[df['TABNAME'] == table].to_csv(f"data/fixed/row_sep/{table}.csv", index=False)

# %%
df = pd.read_csv('data/fixed/table_locks.csv')
df.columns = df.columns.str.strip()
# Remove all trailing whitespace from TABNAME and TABSCHEMA
# NOTE: This seems to be a problem in the table lock data.
df["TABNAME"] = df["TABNAME"].astype(str).apply(lambda x: x.rstrip())
df["TABSCHEMA"] = df["TABSCHEMA"].astype(str).apply(lambda x: x.rstrip())
unique_tables = df[df['TABSCHEMA'] != 'SYSIBM']['TABNAME'].unique()

# %%
unique_tables

# %%
for table in unique_tables:
    print(f"Table: {table}")
    df[df['TABNAME'] == table].to_csv(f"data/fixed/table_sep/{table}.csv", index=False)

# %%



