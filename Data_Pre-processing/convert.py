import pandas as pd
import os
csv_path=('/home/user01/data_ssd/Abbas/Dog1/dog-export.csv')
vott_df = pd.read_csv(csv_path)
#print(vott_df)
labels = vott_df["label"].unique()
#print(labels)
labeldict = dict(zip(labels, range(len(labels))))
vott_df.drop_duplicates(subset=None, keep="first", inplace=True)
#print(labeldict)
path='/home/user01/data_ssd/Abbas/Dog1/'
def convert_vott_csv_to_yolo(
    vott_df,
    labeldict,
    path="",
    target_name="data_train.txt",
    abs_path=False,
):
    # Encode labels according to labeldict if code's don't exist
    if not "code" in vott_df.columns:
        vott_df["code"] = vott_df["label"].apply(lambda x: labeldict[x])
    # Round float to ints
    for col in vott_df[["xmin", "ymin", "xmax", "ymax"]]:
        vott_df[col] = (vott_df[col]).apply(lambda x: round(x))
    # Create Yolo Text file
    last_image = ""
    txt_file = ""
    for index, row in vott_df.iterrows():
        if not last_image == row["image"]:
            if abs_path:
                txt_file += "\n" + row["image_path"] + " "
            else:
                txt_file += "\n" + os.path.join(path, row["image"]) + " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        else:
            txt_file += " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        last_image = row["image"]
    file = open(target_name, "w")
    file.write(txt_file[1:])
    file.close()
    return True
_=convert_vott_csv_to_yolo(vott_df,labeldict,path)
