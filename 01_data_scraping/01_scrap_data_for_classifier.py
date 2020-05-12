'''
Script to scrap data for training meme classifier

Steps:
1. Scrap meme titles from imgflip.com
	- Get the template titles and image urls
2. Download image URLs scarpped from imgflip

'''

from selenium import webdriver
import time
import pandas as pd
import os

#----------------------------------------------------------
#			1. Scrap meme titles from imgflip.com
#----------------------------------------------------------

driver = webdriver.Chrome('/usr/bin/chromedriver')
page_range = range(1, 16)
template_name_list = []
template_df_list = []

for each_page in page_range:
	driver.get('https://imgflip.com/memetemplates?page={}'.format(each_page))

	items = driver.find_elements_by_class_name("mt-boxes")

	total = []

	for item in items:
		a_ele = item.find_elements_by_xpath('//*[@class="mt-boxes"]/div/h3/a')
		img_ele = item.find_elements_by_xpath('//*[@class="mt-boxes"]/div/div/a/img')
		for title, link in zip(a_ele, img_ele):
			template_name_list.append([title.text])
			template_df_list.append([title.text, link.get_attribute("src")])
			# print(each.text)

	time.sleep(5)

template_names_df = pd.DataFrame(template_name_list, columns=["template_title"])
template_data_df = pd.DataFrame(template_df_list, columns=["template_title", "template_url"])
print(template_data_df.shape)
template_names_df.to_csv("template_names_df.csv", index=None)
template_data_df.to_csv("template_data_df.csv", index=None)

#----------------------------------------------------------
#			2. Download image URLs scarpped from imgflip
#----------------------------------------------------------
source_csv = pd.read_csv("template_data_df.csv")
download_dir = "meme_templates_imgflip"
os.makedirs(download_dir, exist_ok=True)
for index, each_row in source_csv.iterrows():
	print(each_row[1])
	new_img_name = each_row[0].lower().replace(" ", "_")
	os.system("wget {} -O {}/{}.jpg".format(each_row[1], download_dir, new_img_name))

print("COMPLETED!!")