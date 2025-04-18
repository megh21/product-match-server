import pandas as pd
import os
import shutil
df = pd.read_csv("data/styles.csv")
#copy files of name 10000 to 10005 from images folder to small_images folder    
#create folder if exists make it empty

shutil.rmtree("data/small_images", ignore_errors=True)
os.makedirs("data/small_images")
for i in range(10000, 10002):
    shutil.copyfile(f"data/images/{i}.jpg", f"data/small_images/{i}.jpg")

#take rows with ids of .png files in data/small_images
ids = [int(x.split(".")[0]) for x in os.listdir("data/small_images")]
df = df[df.id.isin(ids)]
#sort with id
df = df.sort_values("id")
#save to data/styles_small.csv
df.to_csv("data/styles_small.csv", index=False)