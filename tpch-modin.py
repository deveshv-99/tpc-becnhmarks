import argparse
import functools
import inspect
import json
import time
from typing import Callable, List, Dict
import csv
import modin.pandas as pd
import ray

ray.init()
file = open('execution_times.csv', 'a', newline='')
writer = csv.writer(file)

@functools.lru_cache
def load_lineitem(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/lineitem.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    df["l_shipdate"] = pd.to_datetime(df.l_shipdate, format="%Y-%m-%d")
    df["l_receiptdate"] = pd.to_datetime(df.l_receiptdate, format="%Y-%m-%d")
    df["l_commitdate"] = pd.to_datetime(df.l_commitdate, format="%Y-%m-%d")
    return df


@functools.lru_cache
def load_part(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/part.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    return df


@functools.lru_cache
def load_orders(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/orders.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    df["o_orderdate"] = pd.to_datetime(df.o_orderdate, format="%Y-%m-%d")
    return df


@functools.lru_cache
def load_customer(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/customer.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    return df


@functools.lru_cache
def load_nation(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/nation.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    return df


@functools.lru_cache
def load_region(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/region.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    return df


@functools.lru_cache
def load_supplier(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/supplier.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    return df


@functools.lru_cache
def load_partsupp(
    data_folder: str, **storage_options
) -> pd.DataFrame:
    data_path = data_folder + "/partsupp.parquet"
    df = pd.read_parquet(
        data_path,  storage_options=storage_options
    )
    return df


def timethis(q: Callable):
    @functools.wraps(q)
    def wrapped(*args, **kwargs):
        t = time.time()
        q(*args, **kwargs)
        #print("%s Execution time (s): %f" % (q.__name__.upper(), time.time() - t))
        writer.writerow([f"{q.__name__.upper()} ", time.time() - t])
    return wrapped


_query_to_datasets: Dict[int, List[str]] = dict()


def collect_datasets(func: Callable):
    _query_to_datasets[int(func.__name__[1:])] = list(inspect.signature(func).parameters)
    return func


@timethis
@collect_datasets
def q01(lineitem: pd.DataFrame):
    date = pd.Timestamp("1998-09-02")
    lineitem_filtered = lineitem.loc[
        :,
        [
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_orderkey",
        ],
    ]
    sel = lineitem_filtered.l_shipdate <= date
    lineitem_filtered = lineitem_filtered[sel]
    lineitem_filtered["AVG_QTY"] = lineitem_filtered.l_quantity
    lineitem_filtered["AVG_PRICE"] = lineitem_filtered.l_extendedprice
    lineitem_filtered["DISC_PRICE"] = lineitem_filtered.l_extendedprice * (
        1 - lineitem_filtered.l_discount
    )
    lineitem_filtered["CHARGE"] = (
        lineitem_filtered.l_extendedprice
        * (1 - lineitem_filtered.l_discount)
        * (1 + lineitem_filtered.l_tax)
    )
    gb = lineitem_filtered.groupby(["l_returnflag", "l_linestatus"], as_index=False)[
        [
            "l_quantity",
            "l_extendedprice",
            "DISC_PRICE",
            "CHARGE",
            "AVG_QTY",
            "AVG_PRICE",
            "l_discount",
            "l_orderkey",
        ]
    ]
    total = gb.agg(
        {
            "l_quantity": "sum",
            "l_extendedprice": "sum",
            "DISC_PRICE": "sum",
            "CHARGE": "sum",
            "AVG_QTY": "mean",
            "AVG_PRICE": "mean",
            "l_discount": "mean",
            "l_orderkey": "count",
        }
    )
    # skip sort, Mars groupby enables sort
    # total = total.sort_values(["l_returnflag", "l_linestatus"])

    
    ## print(total)


@timethis
@collect_datasets
def q02(part, partsupp, supplier, nation, region):
    nation_filtered = nation.loc[:, ["n_nationkey", "n_name", "n_regionkey"]]
    region_filtered = region[(region["r_name"] == "EUROPE")]
    region_filtered = region_filtered.loc[:, ["r_regionkey"]]
    r_n_merged = nation_filtered.merge(
        region_filtered, left_on="n_regionkey", right_on="r_regionkey", how="inner"
    )
    r_n_merged = r_n_merged.loc[:, ["n_nationkey", "n_name"]]
    supplier_filtered = supplier.loc[
        :,
        [
            "s_suppkey",
            "s_name",
            "s_address",
            "s_nationkey",
            "s_phone",
            "s_acctbal",
            "s_comment",
        ],
    ]
    s_r_n_merged = r_n_merged.merge(
        supplier_filtered, left_on="n_nationkey", right_on="s_nationkey", how="inner"
    )
    s_r_n_merged = s_r_n_merged.loc[
        :,
        [
            "n_name",
            "s_suppkey",
            "s_name",
            "s_address",
            "s_phone",
            "s_acctbal",
            "s_comment",
        ],
    ]
    partsupp_filtered = partsupp.loc[:, ["ps_partkey", "ps_suppkey", "ps_supplycost"]]
    ps_s_r_n_merged = s_r_n_merged.merge(
        partsupp_filtered, left_on="s_suppkey", right_on="ps_suppkey", how="inner"
    )
    ps_s_r_n_merged = ps_s_r_n_merged.loc[
        :,
        [
            "n_name",
            "s_name",
            "s_address",
            "s_phone",
            "s_acctbal",
            "s_comment",
            "ps_partkey",
            "ps_supplycost",
        ],
    ]
    part_filtered = part.loc[:, ["p_partkey", "p_mfgr", "p_size", "p_type"]]
    part_filtered = part_filtered[
        (part_filtered["p_size"] == 15)
        & (part_filtered["p_type"].str.endswith("BRASS"))
    ]
    part_filtered = part_filtered.loc[:, ["p_partkey", "p_mfgr"]]
    merged_df = part_filtered.merge(
        ps_s_r_n_merged, left_on="p_partkey", right_on="ps_partkey", how="inner"
    )
    merged_df = merged_df.loc[
        :,
        [
            "n_name",
            "s_name",
            "s_address",
            "s_phone",
            "s_acctbal",
            "s_comment",
            "ps_supplycost",
            "p_partkey",
            "p_mfgr",
        ],
    ]
    min_values = merged_df.groupby("p_partkey", as_index=False, sort=False)[
        "ps_supplycost"
    ].min()
    min_values.columns = ["p_partkey_CPY", "MIN_SUPPLYCOST"]
    merged_df = merged_df.merge(
        min_values,
        left_on=["p_partkey", "ps_supplycost"],
        right_on=["p_partkey_CPY", "MIN_SUPPLYCOST"],
        how="inner",
    )
    total = merged_df.loc[
        :,
        [
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        ],
    ]
    total = total.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    )
    ##print(total)


@timethis
@collect_datasets
def q03(lineitem, orders, customer):
    date = pd.Timestamp("1995-03-04")
    lineitem_filtered = lineitem.loc[
        :, ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"]
    ]
    orders_filtered = orders.loc[
        :, ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"]
    ]
    customer_filtered = customer.loc[:, ["c_mktsegment", "c_custkey"]]
    lsel = lineitem_filtered.l_shipdate > date
    osel = orders_filtered.o_orderdate < date
    csel = customer_filtered.c_mktsegment == "BUILDING"
    flineitem = lineitem_filtered[lsel]
    forders = orders_filtered[osel]
    fcustomer = customer_filtered[csel]
    jn1 = fcustomer.merge(forders, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(flineitem, left_on="o_orderkey", right_on="l_orderkey")
    jn2["TMP"] = jn2.l_extendedprice * (1 - jn2.l_discount)
    total = (
        jn2.groupby(
            ["l_orderkey", "o_orderdate", "o_shippriority"], as_index=False, sort=False
        )["TMP"]
        .sum()
        .sort_values(["TMP"], ascending=False)
    )
    res = total.loc[:, ["l_orderkey", "TMP", "o_orderdate", "o_shippriority"]]


    #print(res.head(10))


@timethis
@collect_datasets
def q04(lineitem, orders):
    date1 = pd.Timestamp("1993-11-01")
    date2 = pd.Timestamp("1993-08-01")
    lsel = lineitem.l_commitdate < lineitem.l_receiptdate
    osel = (orders.o_orderdate < date1) & (orders.o_orderdate >= date2)
    flineitem = lineitem[lsel]
    forders = orders[osel]
    jn = forders[forders["o_orderkey"].isin(flineitem["l_orderkey"])]
    total = (
        jn.groupby("o_orderpriority", as_index=False)["o_orderkey"].count()
        # skip sort when Mars enables sort in groupby
        # .sort_values(["o_orderpriority"])
    )

    #print(total)


@timethis
@collect_datasets
def q05(lineitem, orders, customer, nation, region, supplier):
    date1 = pd.Timestamp("1996-01-01")
    date2 = pd.Timestamp("1997-01-01")
    rsel = region.r_name == "ASIA"
    osel = (orders.o_orderdate >= date1) & (orders.o_orderdate < date2)
    forders = orders[osel]
    fregion = region[rsel]
    jn1 = fregion.merge(nation, left_on="r_regionkey", right_on="n_regionkey")
    jn2 = jn1.merge(customer, left_on="n_nationkey", right_on="c_nationkey")
    jn3 = jn2.merge(forders, left_on="c_custkey", right_on="o_custkey")
    jn4 = jn3.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
    jn5 = supplier.merge(
        jn4, left_on=["s_suppkey", "s_nationkey"], right_on=["l_suppkey", "n_nationkey"]
    )
    jn5["TMP"] = jn5.l_extendedprice * (1.0 - jn5.l_discount)
    gb = jn5.groupby("n_name", as_index=False, sort=False)["TMP"].sum()
    total = gb.sort_values("TMP", ascending=False)

    #print(total)


@timethis
@collect_datasets
def q06(lineitem):
    date1 = pd.Timestamp("1996-01-01")
    date2 = pd.Timestamp("1997-01-01")
    lineitem_filtered = lineitem.loc[
        :, ["l_quantity", "l_extendedprice", "l_discount", "l_shipdate"]
    ]
    sel = (
        (lineitem_filtered.l_shipdate >= date1)
        & (lineitem_filtered.l_shipdate < date2)
        & (lineitem_filtered.l_discount >= 0.08)
        & (lineitem_filtered.l_discount <= 0.1)
        & (lineitem_filtered.l_quantity < 24)
    )
    flineitem = lineitem_filtered[sel]
    total = (flineitem.l_extendedprice * flineitem.l_discount).sum()

    #print(total)


@timethis
@collect_datasets
def q07(lineitem, supplier, orders, customer, nation):
    """This version is faster than q07_old. Keeping the old one for reference"""
    lineitem_filtered = lineitem[
        (lineitem["l_shipdate"] >= pd.Timestamp("1995-01-01"))
        & (lineitem["l_shipdate"] < pd.Timestamp("1997-01-01"))
    ]
    lineitem_filtered["l_year"] = lineitem_filtered["l_shipdate"].dt.year
    lineitem_filtered["VOLUME"] = lineitem_filtered["l_extendedprice"] * (
        1.0 - lineitem_filtered["l_discount"]
    )
    lineitem_filtered = lineitem_filtered.loc[
        :, ["l_orderkey", "l_suppkey", "l_year", "VOLUME"]
    ]
    supplier_filtered = supplier.loc[:, ["s_suppkey", "s_nationkey"]]
    orders_filtered = orders.loc[:, ["o_orderkey", "o_custkey"]]
    customer_filtered = customer.loc[:, ["c_custkey", "c_nationkey"]]
    n1 = nation[(nation["n_name"] == "FRANCE")].loc[:, ["n_nationkey", "n_name"]]
    n2 = nation[(nation["n_name"] == "GERMANY")].loc[:, ["n_nationkey", "n_name"]]

    # ----- do nation 1 -----
    N1_C = customer_filtered.merge(
        n1, left_on="c_nationkey", right_on="n_nationkey", how="inner"
    )
    N1_C = N1_C.drop(columns=["c_nationkey", "n_nationkey"]).rename(
        columns={"n_name": "cust_nation"}
    )
    N1_C_O = N1_C.merge(
        orders_filtered, left_on="c_custkey", right_on="o_custkey", how="inner"
    )
    N1_C_O = N1_C_O.drop(columns=["c_custkey", "o_custkey"])

    N2_S = supplier_filtered.merge(
        n2, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    )
    N2_S = N2_S.drop(columns=["s_nationkey", "n_nationkey"]).rename(
        columns={"n_name": "supp_nation"}
    )
    N2_S_L = N2_S.merge(
        lineitem_filtered, left_on="s_suppkey", right_on="l_suppkey", how="inner"
    )
    N2_S_L = N2_S_L.drop(columns=["s_suppkey", "l_suppkey"])

    total1 = N1_C_O.merge(
        N2_S_L, left_on="o_orderkey", right_on="l_orderkey", how="inner"
    )
    total1 = total1.drop(columns=["o_orderkey", "l_orderkey"])

    # ----- do nation 2 -----
    N2_C = customer_filtered.merge(
        n2, left_on="c_nationkey", right_on="n_nationkey", how="inner"
    )
    N2_C = N2_C.drop(columns=["c_nationkey", "n_nationkey"]).rename(
        columns={"n_name": "cust_nation"}
    )
    N2_C_O = N2_C.merge(
        orders_filtered, left_on="c_custkey", right_on="o_custkey", how="inner"
    )
    N2_C_O = N2_C_O.drop(columns=["c_custkey", "o_custkey"])

    N1_S = supplier_filtered.merge(
        n1, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    )
    N1_S = N1_S.drop(columns=["s_nationkey", "n_nationkey"]).rename(
        columns={"n_name": "supp_nation"}
    )
    N1_S_L = N1_S.merge(
        lineitem_filtered, left_on="s_suppkey", right_on="l_suppkey", how="inner"
    )
    N1_S_L = N1_S_L.drop(columns=["s_suppkey", "l_suppkey"])

    total2 = N2_C_O.merge(
        N1_S_L, left_on="o_orderkey", right_on="l_orderkey", how="inner"
    )
    total2 = total2.drop(columns=["o_orderkey", "l_orderkey"])

    # concat results
    total = pd.concat([total1, total2])

    total = total.groupby(["supp_nation", "cust_nation", "l_year"], as_index=False).agg(
        REVENUE=pd.NamedAgg(column="VOLUME", aggfunc="sum")
    )
    # skip sort when Mars groupby does sort already
    # total = total.sort_values(
    #     by=["supp_nation", "cust_nation", "l_year"], ascending=[True, True, True]
    # )

    #print(total)


@timethis
@collect_datasets
def q08(part, lineitem, supplier, orders, customer, nation, region):
    part_filtered = part[(part["p_type"] == "ECONOMY ANODIZED STEEL")]
    part_filtered = part_filtered.loc[:, ["p_partkey"]]
    lineitem_filtered = lineitem.loc[:, ["l_partkey", "l_suppkey", "l_orderkey"]]
    lineitem_filtered["VOLUME"] = lineitem["l_extendedprice"] * (
        1.0 - lineitem["l_discount"]
    )
    total = part_filtered.merge(
        lineitem_filtered, left_on="p_partkey", right_on="l_partkey", how="inner"
    )
    total = total.loc[:, ["l_suppkey", "l_orderkey", "VOLUME"]]
    supplier_filtered = supplier.loc[:, ["s_suppkey", "s_nationkey"]]
    total = total.merge(
        supplier_filtered, left_on="l_suppkey", right_on="s_suppkey", how="inner"
    )
    total = total.loc[:, ["l_orderkey", "VOLUME", "s_nationkey"]]
    orders_filtered = orders[
        (orders["o_orderdate"] >= pd.Timestamp("1995-01-01"))
        & (orders["o_orderdate"] < pd.Timestamp("1997-01-01"))
    ]
    orders_filtered["o_year"] = orders_filtered["o_orderdate"].dt.year
    orders_filtered = orders_filtered.loc[:, ["o_orderkey", "o_custkey", "o_year"]]
    total = total.merge(
        orders_filtered, left_on="l_orderkey", right_on="o_orderkey", how="inner"
    )
    total = total.loc[:, ["VOLUME", "s_nationkey", "o_custkey", "o_year"]]
    customer_filtered = customer.loc[:, ["c_custkey", "c_nationkey"]]
    total = total.merge(
        customer_filtered, left_on="o_custkey", right_on="c_custkey", how="inner"
    )
    total = total.loc[:, ["VOLUME", "s_nationkey", "o_year", "c_nationkey"]]
    n1_filtered = nation.loc[:, ["n_nationkey", "n_regionkey"]]
    n2_filtered = nation.loc[:, ["n_nationkey", "n_name"]].rename(
        columns={"n_name": "nation"}
    )
    total = total.merge(
        n1_filtered, left_on="c_nationkey", right_on="n_nationkey", how="inner"
    )
    total = total.loc[:, ["VOLUME", "s_nationkey", "o_year", "n_regionkey"]]
    total = total.merge(
        n2_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    )
    total = total.loc[:, ["VOLUME", "o_year", "n_regionkey", "nation"]]
    region_filtered = region[(region["r_name"] == "AMERICA")]
    region_filtered = region_filtered.loc[:, ["r_regionkey"]]
    total = total.merge(
        region_filtered, left_on="n_regionkey", right_on="r_regionkey", how="inner"
    )
    total = total.loc[:, ["VOLUME", "o_year", "nation"]]

    def udf(df):
        demonimator = df["VOLUME"].sum()
        df = df[df["nation"] == "BRAZIL"]
        numerator = df["VOLUME"].sum()
        return numerator / demonimator

    total = total.groupby("o_year", as_index=False).apply(udf)
    total.columns = ["o_year", "mkt_share"]
    total = total.sort_values(by=["o_year"], ascending=[True])

    #print(total)


@timethis
@collect_datasets
def q09(lineitem, orders, part, nation, partsupp, supplier):
    psel = part.p_name.str.contains("ghost")
    fpart = part[psel]
    jn1 = lineitem.merge(fpart, left_on="l_partkey", right_on="p_partkey")
    jn2 = jn1.merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
    jn3 = jn2.merge(nation, left_on="s_nationkey", right_on="n_nationkey")
    jn4 = partsupp.merge(
        jn3, left_on=["ps_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"]
    )
    jn5 = jn4.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    jn5["TMP"] = jn5.l_extendedprice * (1 - jn5.l_discount) - (
        (1 * jn5.ps_supplycost) * jn5.l_quantity
    )
    jn5["o_year"] = jn5.o_orderdate.dt.year
    gb = jn5.groupby(["n_name", "o_year"], as_index=False, sort=False)["TMP"].sum()
    total = gb.sort_values(["n_name", "o_year"], ascending=[True, False])

    #print(total)


@timethis
@collect_datasets
def q10(lineitem, orders, customer, nation):
    date1 = pd.Timestamp("1994-11-01")
    date2 = pd.Timestamp("1995-02-01")
    osel = (orders.o_orderdate >= date1) & (orders.o_orderdate < date2)
    lsel = lineitem.l_returnflag == "R"
    forders = orders[osel]
    flineitem = lineitem[lsel]
    jn1 = flineitem.merge(forders, left_on="l_orderkey", right_on="o_orderkey")
    jn2 = jn1.merge(customer, left_on="o_custkey", right_on="c_custkey")
    jn3 = jn2.merge(nation, left_on="c_nationkey", right_on="n_nationkey")
    jn3["TMP"] = jn3.l_extendedprice * (1.0 - jn3.l_discount)
    gb = jn3.groupby(
        [
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        ],
        as_index=False,
        sort=False,
    )["TMP"].sum()
    total = gb.sort_values("TMP", ascending=False)

    #print(total.head(20))


@timethis
@collect_datasets
def q11(partsupp, supplier, nation):
    partsupp_filtered = partsupp.loc[:, ["ps_partkey", "ps_suppkey"]]
    partsupp_filtered["TOTAL_COST"] = (
        partsupp["ps_supplycost"] * partsupp["ps_availqty"]
    )
    supplier_filtered = supplier.loc[:, ["s_suppkey", "s_nationkey"]]
    ps_supp_merge = partsupp_filtered.merge(
        supplier_filtered, left_on="ps_suppkey", right_on="s_suppkey", how="inner"
    )
    ps_supp_merge = ps_supp_merge.loc[:, ["ps_partkey", "s_nationkey", "TOTAL_COST"]]
    nation_filtered = nation[(nation["n_name"] == "GERMANY")]
    nation_filtered = nation_filtered.loc[:, ["n_nationkey"]]
    ps_supp_n_merge = ps_supp_merge.merge(
        nation_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    )
    ps_supp_n_merge = ps_supp_n_merge.loc[:, ["ps_partkey", "TOTAL_COST"]]
    sum_val = ps_supp_n_merge["TOTAL_COST"].sum() * 0.0001
    total = ps_supp_n_merge.groupby(["ps_partkey"], as_index=False, sort=False).agg(
        VALUE=pd.NamedAgg(column="TOTAL_COST", aggfunc="sum")
    )
    total = total[total["VALUE"] > sum_val]
    total = total.sort_values("VALUE", ascending=False)

    #print(total)


@timethis
@collect_datasets
def q12(lineitem, orders):
    date1 = pd.Timestamp("1994-01-01")
    date2 = pd.Timestamp("1995-01-01")
    sel = (
        (lineitem.l_receiptdate < date2)
        & (lineitem.l_commitdate < date2)
        & (lineitem.l_shipdate < date2)
        & (lineitem.l_shipdate < lineitem.l_commitdate)
        & (lineitem.l_commitdate < lineitem.l_receiptdate)
        & (lineitem.l_receiptdate >= date1)
        & ((lineitem.l_shipmode == "MAIL") | (lineitem.l_shipmode == "SHIP"))
    )
    flineitem = lineitem[sel]
    jn = flineitem.merge(orders, left_on="l_orderkey", right_on="o_orderkey")

    def g1(x):
        return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

    def g2(x):
        return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

    total = jn.groupby("l_shipmode", as_index=False)["o_orderpriority"].agg((g1, g2))
    total = total.reset_index()  # reset index to keep consistency with pandas
    # skip sort when groupby does sort already
    # total = total.sort_values("l_shipmode")
    
    #print(total)


@timethis
@collect_datasets
def q13(customer, orders):
    return
    customer_filtered = customer.loc[:, ["c_custkey"]]
    orders_filtered = orders[
        ~orders["o_comment"].str.contains(r"special[\S|\s]*requests")
    ]
    orders_filtered = orders_filtered.loc[:, ["o_orderkey", "o_custkey"]]
    c_o_merged = customer_filtered.merge(
        orders_filtered, left_on="c_custkey", right_on="o_custkey", how="left"
    )
    c_o_merged = c_o_merged.loc[:, ["c_custkey", "o_orderkey"]]
    count_df = c_o_merged.groupby(["c_custkey"], as_index=False, sort=False).agg(
        c_count=pd.NamedAgg(column="o_orderkey", aggfunc="count")
    )
    total = count_df.groupby(["c_count"], as_index=False, sort=False).size()
    total.columns = ["c_count", "CUSTDIST"]
    total = total.sort_values(by=["CUSTDIST", "c_count"], ascending=[False, False])
    
    #print(total)


@timethis
@collect_datasets
def q14(lineitem, part):
    startDate = pd.Timestamp("1994-03-01")
    endDate = pd.Timestamp("1994-04-01")
    p_type_like = "PROMO"
    part_filtered = part.loc[:, ["p_partkey", "p_type"]]
    lineitem_filtered = lineitem.loc[
        :, ["l_extendedprice", "l_discount", "l_shipdate", "l_partkey"]
    ]
    sel = (lineitem_filtered.l_shipdate >= startDate) & (
        lineitem_filtered.l_shipdate < endDate
    )
    flineitem = lineitem_filtered[sel]
    jn = flineitem.merge(part_filtered, left_on="l_partkey", right_on="p_partkey")
    jn["TMP"] = jn.l_extendedprice * (1.0 - jn.l_discount)
    total = jn[jn.p_type.str.startswith(p_type_like)].TMP.sum() * 100 / jn.TMP.sum()
    
    #print(total)


@timethis
@collect_datasets
def q15(lineitem, supplier):
    lineitem_filtered = lineitem[
        (lineitem["l_shipdate"] >= pd.Timestamp("1996-01-01"))
        & (
            lineitem["l_shipdate"]
            < (pd.Timestamp("1996-01-01") + pd.DateOffset(months=3))
        )
    ]
    lineitem_filtered["REVENUE_PARTS"] = lineitem_filtered["l_extendedprice"] * (
        1.0 - lineitem_filtered["l_discount"]
    )
    lineitem_filtered = lineitem_filtered.loc[:, ["l_suppkey", "REVENUE_PARTS"]]
    revenue_table = (
        lineitem_filtered.groupby("l_suppkey", as_index=False, sort=False)
        .agg(TOTAL_REVENUE=pd.NamedAgg(column="REVENUE_PARTS", aggfunc="sum"))
        .rename(columns={"l_suppkey": "SUPPLIER_NO"})
    )
    max_revenue = revenue_table["TOTAL_REVENUE"].max()
    revenue_table = revenue_table[revenue_table["TOTAL_REVENUE"] == max_revenue]
    supplier_filtered = supplier.loc[:, ["s_suppkey", "s_name", "s_address", "s_phone"]]
    total = supplier_filtered.merge(
        revenue_table, left_on="s_suppkey", right_on="SUPPLIER_NO", how="inner"
    )
    total = total.loc[
        :, ["s_suppkey", "s_name", "s_address", "s_phone", "TOTAL_REVENUE"]
    ]
    #print(total)


@timethis
@collect_datasets
def q16(part, partsupp, supplier):
    part_filtered = part[
        (part["p_brand"] != "Brand#45")
        & (~part["p_type"].str.contains("^MEDIUM POLISHED"))
        & part["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])
    ]
    part_filtered = part_filtered.loc[:, ["p_partkey", "p_brand", "p_type", "p_size"]]
    partsupp_filtered = partsupp.loc[:, ["ps_partkey", "ps_suppkey"]]
    total = part_filtered.merge(
        partsupp_filtered, left_on="p_partkey", right_on="ps_partkey", how="inner"
    )
    total = total.loc[:, ["p_brand", "p_type", "p_size", "ps_suppkey"]]
    supplier_filtered = supplier[
        supplier["s_comment"].str.contains(r"Customer(\S|\s)*Complaints")
    ]
    supplier_filtered = supplier_filtered.loc[:, ["s_suppkey"]].drop_duplicates()
    # left merge to select only ps_suppkey values not in supplier_filtered
    total = total.merge(
        supplier_filtered, left_on="ps_suppkey", right_on="s_suppkey", how="left"
    )
    total = total[total["s_suppkey"].isna()]
    total = total.loc[:, ["p_brand", "p_type", "p_size", "ps_suppkey"]]
    total = total.groupby(["p_brand", "p_type", "p_size"], as_index=False, sort=False)[
        "ps_suppkey"
    ].nunique()
    total.columns = ["p_brand", "p_type", "p_size", "SUPPLIER_CNT"]
    total = total.sort_values(
        by=["SUPPLIER_CNT", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    )

    #print(total)


@timethis
@collect_datasets
def q17(lineitem, part):
    left = lineitem.loc[:, ["l_partkey", "l_quantity", "l_extendedprice"]]
    right = part[((part["p_brand"] == "Brand#23") & (part["p_container"] == "MED BOX"))]
    right = right.loc[:, ["p_partkey"]]
    line_part_merge = left.merge(
        right, left_on="l_partkey", right_on="p_partkey", how="inner"
    )
    line_part_merge = line_part_merge.loc[
        :, ["l_quantity", "l_extendedprice", "p_partkey"]
    ]
    lineitem_filtered = lineitem.loc[:, ["l_partkey", "l_quantity"]]
    lineitem_avg = lineitem_filtered.groupby(
        ["l_partkey"], as_index=False, sort=False
    ).agg(avg=pd.NamedAgg(column="l_quantity", aggfunc="mean"))
    lineitem_avg["avg"] = 0.2 * lineitem_avg["avg"]
    lineitem_avg = lineitem_avg.loc[:, ["l_partkey", "avg"]]
    total = line_part_merge.merge(
        lineitem_avg, left_on="p_partkey", right_on="l_partkey", how="inner"
    )
    total = total[total["l_quantity"] < total["avg"]]
    total = pd.DataFrame({"avg_yearly": [total["l_extendedprice"].sum() / 7.0]})
    
    #print(total)


@timethis
@collect_datasets
def q18(lineitem, orders, customer):
    gb1 = lineitem.groupby("l_orderkey", as_index=False, sort=False)["l_quantity"].sum()
    fgb1 = gb1[gb1.l_quantity > 300]
    jn1 = fgb1.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    jn2 = jn1.merge(customer, left_on="o_custkey", right_on="c_custkey")
    gb2 = jn2.groupby(
        ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"],
        as_index=False,
        sort=False,
    )["l_quantity"].sum()
    total = gb2.sort_values(["o_totalprice", "o_orderdate"], ascending=[False, True])
    
    #print(total.head(100))


@timethis
@collect_datasets
def q19(lineitem, part):
    Brand31 = "Brand#31"
    Brand43 = "Brand#43"
    SMBOX = "SM BOX"
    SMCASE = "SM CASE"
    SMPACK = "SM PACK"
    SMPKG = "SM PKG"
    MEDBAG = "MED BAG"
    MEDBOX = "MED BOX"
    MEDPACK = "MED PACK"
    MEDPKG = "MED PKG"
    LGBOX = "LG BOX"
    LGCASE = "LG CASE"
    LGPACK = "LG PACK"
    LGPKG = "LG PKG"
    DELIVERINPERSON = "DELIVER IN PERSON"
    AIR = "AIR"
    AIRREG = "AIRREG"
    lsel = (
        (
            ((lineitem.l_quantity <= 36) & (lineitem.l_quantity >= 26))
            | ((lineitem.l_quantity <= 25) & (lineitem.l_quantity >= 15))
            | ((lineitem.l_quantity <= 14) & (lineitem.l_quantity >= 4))
        )
        & (lineitem.l_shipinstruct == DELIVERINPERSON)
        & ((lineitem.l_shipmode == AIR) | (lineitem.l_shipmode == AIRREG))
    )
    psel = (part.p_size >= 1) & (
        (
            (part.p_size <= 5)
            & (part.p_brand == Brand31)
            & (
                (part.p_container == SMBOX)
                | (part.p_container == SMCASE)
                | (part.p_container == SMPACK)
                | (part.p_container == SMPKG)
            )
        )
        | (
            (part.p_size <= 10)
            & (part.p_brand == Brand43)
            & (
                (part.p_container == MEDBAG)
                | (part.p_container == MEDBOX)
                | (part.p_container == MEDPACK)
                | (part.p_container == MEDPKG)
            )
        )
        | (
            (part.p_size <= 15)
            & (part.p_brand == Brand43)
            & (
                (part.p_container == LGBOX)
                | (part.p_container == LGCASE)
                | (part.p_container == LGPACK)
                | (part.p_container == LGPKG)
            )
        )
    )
    flineitem = lineitem[lsel]
    fpart = part[psel]
    jn = flineitem.merge(fpart, left_on="l_partkey", right_on="p_partkey")
    jnsel = (
        (jn.p_brand == Brand31)
        & (
            (jn.p_container == SMBOX)
            | (jn.p_container == SMCASE)
            | (jn.p_container == SMPACK)
            | (jn.p_container == SMPKG)
        )
        & (jn.l_quantity >= 4)
        & (jn.l_quantity <= 14)
        & (jn.p_size <= 5)
        | (jn.p_brand == Brand43)
        & (
            (jn.p_container == MEDBAG)
            | (jn.p_container == MEDBOX)
            | (jn.p_container == MEDPACK)
            | (jn.p_container == MEDPKG)
        )
        & (jn.l_quantity >= 15)
        & (jn.l_quantity <= 25)
        & (jn.p_size <= 10)
        | (jn.p_brand == Brand43)
        & (
            (jn.p_container == LGBOX)
            | (jn.p_container == LGCASE)
            | (jn.p_container == LGPACK)
            | (jn.p_container == LGPKG)
        )
        & (jn.l_quantity >= 26)
        & (jn.l_quantity <= 36)
        & (jn.p_size <= 15)
    )
    jn = jn[jnsel]
    total = (jn.l_extendedprice * (1.0 - jn.l_discount)).sum()
    
    #print(total)


@timethis
@collect_datasets
def q20(lineitem, part, nation, partsupp, supplier):
    date1 = pd.Timestamp("1996-01-01")
    date2 = pd.Timestamp("1997-01-01")
    psel = part.p_name.str.startswith("azure")
    nsel = nation.n_name == "JORDAN"
    lsel = (lineitem.l_shipdate >= date1) & (lineitem.l_shipdate < date2)
    fpart = part[psel]
    fnation = nation[nsel]
    flineitem = lineitem[lsel]
    jn1 = fpart.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
    jn2 = jn1.merge(
        flineitem,
        left_on=["ps_partkey", "ps_suppkey"],
        right_on=["l_partkey", "l_suppkey"],
    )
    gb = jn2.groupby(
        ["ps_partkey", "ps_suppkey", "ps_availqty"], as_index=False, sort=False
    )["l_quantity"].sum()
    gbsel = gb.ps_availqty > (0.5 * gb.l_quantity)
    fgb = gb[gbsel]
    jn3 = fgb.merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
    jn4 = fnation.merge(jn3, left_on="n_nationkey", right_on="s_nationkey")
    jn4 = jn4.loc[:, ["s_name", "s_address"]]
    total = jn4.sort_values("s_name").drop_duplicates()
    
    #print(total)


@timethis
@collect_datasets
def q21(lineitem, orders, supplier, nation):
    lineitem_filtered = lineitem.loc[
        :, ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]
    ]

    # Keep all rows that have another row in linetiem with the same orderkey and different suppkey
    lineitem_orderkeys = (
        lineitem_filtered.loc[:, ["l_orderkey", "l_suppkey"]]
        .groupby("l_orderkey", as_index=False, sort=False)["l_suppkey"]
        .nunique()
    )
    lineitem_orderkeys.columns = ["l_orderkey", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] > 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["l_orderkey"]]

    # Keep all rows that have l_receiptdate > l_commitdate
    lineitem_filtered = lineitem_filtered[
        lineitem_filtered["l_receiptdate"] > lineitem_filtered["l_commitdate"]
    ]
    lineitem_filtered = lineitem_filtered.loc[:, ["l_orderkey", "l_suppkey"]]

    # Merge Filter + Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="l_orderkey", how="inner"
    )

    # Not Exists: Check the exists condition isn't still satisfied on the output.
    lineitem_orderkeys = lineitem_filtered.groupby(
        "l_orderkey", as_index=False, sort=False
    )["l_suppkey"].nunique()
    lineitem_orderkeys.columns = ["l_orderkey", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] == 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["l_orderkey"]]

    # Merge Filter + Not Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="l_orderkey", how="inner"
    )

    orders_filtered = orders.loc[:, ["o_orderstatus", "o_orderkey"]]
    orders_filtered = orders_filtered[orders_filtered["o_orderstatus"] == "F"]
    orders_filtered = orders_filtered.loc[:, ["o_orderkey"]]
    total = lineitem_filtered.merge(
        orders_filtered, left_on="l_orderkey", right_on="o_orderkey", how="inner"
    )
    total = total.loc[:, ["l_suppkey"]]

    supplier_filtered = supplier.loc[:, ["s_suppkey", "s_nationkey", "s_name"]]
    total = total.merge(
        supplier_filtered, left_on="l_suppkey", right_on="s_suppkey", how="inner"
    )
    total = total.loc[:, ["s_nationkey", "s_name"]]
    nation_filtered = nation.loc[:, ["n_name", "n_nationkey"]]
    nation_filtered = nation_filtered[nation_filtered["n_name"] == "SAUDI ARABIA"]
    total = total.merge(
        nation_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    )
    total = total.loc[:, ["s_name"]]
    total = total.groupby("s_name", as_index=False, sort=False).size()
    total.columns = ["s_name", "NUMWAIT"]
    total = total.sort_values(by=["NUMWAIT", "s_name"], ascending=[False, True])
    
    #print(total)


@timethis
@collect_datasets
def q22(customer, orders):
    customer_filtered = customer.loc[:, ["c_acctbal", "c_custkey"]]
    customer_filtered["CNTRYCODE"] = customer["c_phone"].str.slice(0, 2)
    customer_filtered = customer_filtered[
        (customer["c_acctbal"] > 0.00)
        & customer_filtered["CNTRYCODE"].isin(
            ["13", "31", "23", "29", "30", "18", "17"]
        )
    ]
    avg_value = customer_filtered["c_acctbal"].mean()
    customer_filtered = customer_filtered[customer_filtered["c_acctbal"] > avg_value]
    # Select only the keys that don't match by performing a left join and only selecting columns with an na value
    orders_filtered = orders.loc[:, ["o_custkey"]].drop_duplicates()
    customer_keys = customer_filtered.loc[:, ["c_custkey"]].drop_duplicates()
    customer_selected = customer_keys.merge(
        orders_filtered, left_on="c_custkey", right_on="o_custkey", how="left"
    )
    customer_selected = customer_selected[customer_selected["o_custkey"].isna()]
    customer_selected = customer_selected.loc[:, ["c_custkey"]]
    customer_selected = customer_selected.merge(
        customer_filtered, on="c_custkey", how="inner"
    )
    customer_selected = customer_selected.loc[:, ["CNTRYCODE", "c_acctbal"]]
    agg1 = customer_selected.groupby(["CNTRYCODE"], as_index=False, sort=False).size()
    agg1.columns = ["CNTRYCODE", "NUMCUST"]
    agg2 = customer_selected.groupby(["CNTRYCODE"], as_index=False, sort=False).agg(
        TOTACCTBAL=pd.NamedAgg(column="c_acctbal", aggfunc="sum")
    )
    total = agg1.merge(agg2, on="CNTRYCODE", how="inner")
    total = total.sort_values(by=["CNTRYCODE"], ascending=[True])
    
    #print(total)


def run_queries(
    root: str,
    storage_options: Dict[str, str],
    queries: List[int],
):
    total_start = time.time()
    print("Start data loading")
    queries_to_args = dict()
    datasets_to_load = set()
    for query in queries:
        args = []
        for dataset in _query_to_datasets[query]:
            args.append(
                globals()[f"load_{dataset}"](root, **storage_options)
            )
        queries_to_args[query] = args
    print(f"Data loading time (s): {time.time() - total_start}")

    total_start = time.time()
    for query in queries:
        for _ in range(11):
            globals()[f"q{query:02}"](*queries_to_args[query])
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        help="./tables_scale_1",
    )
    parser.add_argument(
        "--storage_options",
        type=str,
        required=False,
        help="Path to the storage options json file.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="Comma separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--pyarrow-dtype",
        default=False,
        action="store_true",
        help="Use arrow dtype.",
    )
    parser.add_argument(
        "--lazy-copy",
        default=False,
        action="store_true",
        help="Use arrow dtype.",
    )


    args = parser.parse_args()
    data_set = args.data_set


    if args.pyarrow_dtype:
        print("Enable pyarrow dtype")
        pd.set_option("mode.dtype_backend", "pyarrow")


    if args.lazy_copy:
        print("Enable lazy copy")
        pd.set_option("mode.copy_on_write", True)

    # credentials to access the datasource.
    storage_options = {}
    if args.storage_options is not None:
        with open(args.storage_options, "r") as fp:
            storage_options = json.load(fp)
    print(f"Storage options: {storage_options}")

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")

    run_queries(
        data_set,
        storage_options=storage_options,
        queries=queries,
    )
    file.close()


if __name__ == "__main__":
    print(f"Running TPC-H against modin v{pd.__version__}")
    main()