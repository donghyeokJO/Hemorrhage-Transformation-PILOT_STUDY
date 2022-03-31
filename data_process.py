import warnings

warnings.filterwarnings(action="ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)


import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def load_excel_data(
    data_path: str = "data/data_sampling_emr.xlsx",
) -> pd.DataFrame:

    dataset = pd.read_excel(data_path)
    dataset = dataset.astype({"hosp_id": "str"})
    dataset = dataset.set_index(["reg_num", "hosp_id"])
    # print(dataset)
    drop_columns = [
        "Study Date",
        "Manufacturer's Model Name",
        "Manufacturer",
        "Slice Thickness",
        "KVP",
        "X-Ray Tube Current",
        "Rows",
        "Columns",
        "dis_mrs",
        "iv_start",
        "ia_start",
        "ia_end",
    ]

    dataset.drop(drop_columns, axis=1, inplace=True)

    return dataset


def load_spss_data(data_path: str = "data/pilot_study_HTf.sav") -> pd.DataFrame:
    dataset = pd.read_spss(data_path)

    drop_columns = [
        "type",
        "HTf",
        "type_sub",
        "male",
        "age",
        "bmi",
        "dis_mrs",
        "ini_nih",
        "pre_mrs",
        "toast",
        "hx_str",
        "hx_htn",
        "hx_dm",
        "smok",
        "hx_af",
        "hx_hl",
        "htx_plt",
        "htx_coa",
        "htx_htn",
        "htx_statin",
        "htx_dm",
        "tx_throm",
        "rtpa",
        "wbc",
        "hb",
        "hct",
        "plt",
        "tc",
        "tg",
        "hdl",
        "ldl",
        "bun",
        "cr",
        "fbs",
        "ha1c",
        "pt",
        "crp",
        "sbp",
        "dbp",
        "filter_$",
        "dis_mrs_3",
        "dis_mrs_2",
        "dis_mrs_4",
        "type2",
        "slice_start",
        "slice_end",
    ]

    dataset = dataset.astype({"hosp_id": "int"})
    dataset = dataset.astype({"hosp_id": "str"})
    dataset = dataset.set_index(["reg_num", "hosp_id"])

    dataset.drop(drop_columns, axis=1, inplace=True)

    return dataset


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    excel_data = load_excel_data()
    spss_data = load_spss_data()

    new_dataset = pd.concat([excel_data, spss_data], axis=1)

    labels = pd.DataFrame(new_dataset.loc[:, ["HTf"]])
    new_dataset.drop(["HTf", "type_sub", "type"], axis=1, inplace=True)
    columns = new_dataset.columns

    scaler = MinMaxScaler()

    new_dataset = pd.DataFrame(
        scaler.fit_transform(new_dataset),
        columns=columns,
        index=new_dataset.index,
    )

    return new_dataset, labels


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    new_dataset, labels = load_data()
    # print(new_dataset.loc[["hosp_id"]])
