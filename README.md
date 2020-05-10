# meme_organiser

Meme manager for cell phones:

1. Classify images as meme or not
2. For all the images that are memes, create meta tags based on:
	1. Identify different meta tags for the images based on:
		- Celebrity present in the meme
	2. Popular meme templates and references
	3. Text present in the meme
		- Topic of conversation in the text
		- Sentiment analysis of the test
3. Store the image path and meta tags in a db
4. Provide an interface to retrieve the meme image from the db based on the meta tags via search

Basic requirements:
1. Entire project should be able to run on cell phone
2. Any kind of image data should not be stored after processing
