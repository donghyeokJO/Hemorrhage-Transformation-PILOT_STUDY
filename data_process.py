import pandas as pd

from typing import Union


def load_data(
    data_path: str = "data/data_sampling_emr.xlsx",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_excel(data_path, index_col=[0, 1])

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
    # print(dataset.columns)
    # labels = pd.DataFrame(dataset.loc[:, ["HTf", "type_sub"]])
    labels = pd.DataFrame(dataset.loc[:, ["HTf"]])
    dataset.drop(["HTf", "type_sub", "type"], axis=1, inplace=True)
    # print(labels)

    return dataset, labels


if __name__ == "__main__":
    data, labels = load_data()
    print(data)
    print(labels)
    # print(processed_data)
