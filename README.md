# pix2para
From Pixel to Paragraph: A Deep Artwork Analysis Paragraph Generator

### Introduction
Art influences society by shaping our perspective and influencing our sense of self. By portraying subject matter with deliberate composition, color scheme, and other stylistic choices, the artist forms emotional connections between artwork and viewer and communicates their vision with the world. The goal of my project is to develop an artificial neural network system that interprets input artwork and generates a passage that describes the objects and other features (ex. color palette) present in the artwork as well as the ideas and emotions the artwork conveys.

### Approach
I experimented with and modified SeqGAN and LeakGAN frameworks (both of which are originally unconditional GANs) to condition on artwork image features to generate the final art analysis paragraph. 

I have used and modified code from the following repositories: 
* [SeqGAN-Tensorflow](https://github.com/ChenChengKuan/SeqGAN_tensorflow)
* [Show_and_Tell](https://github.com/nikhilmaram/Show_and_Tell)
* [img2poem](https://github.com/bei21/img2poem)
* [Conditional-GAN](https://github.com/zhangqianhui/Conditional-GAN)
* [Texygen](https://github.com/geek-ai/Texygen)
* [tensorflow_compact_bilinear_pooling](https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling)
* [tensorflow-seq2seq-tutorials](https://github.com/ematvey/tensorflow-seq2seq-tutorials)

I wrote scripts for scraping artwork images and their corresponding analysis paragraphs from [The Art Story](https://www.theartstory.org/) and [Smithsonian American Art Museum](https://americanart.si.edu/). 

More details about my modifications to existing code can be found as comments in the files.  

### Select Results 
<img src="https://github.com/audreycui/pix2para/blob/master/images/art_desc785.jpg" height="220px" align="left">
<br/>"sexual of took boy singing communist mixture to above and center emphasizing repetition stand melting."
<br/><br/><br/><br/><br/><br/><br/>
<img src="https://github.com/audreycui/pix2para/blob/master/images/art_desc105.jpg" height="220px" align="left">
<br/>"1908 to one most movements geometric is when, major one humanize(avant-garde) behind in created flowers theme distribution is playful and the."
<br/><br/><br/><br/><br/><br/><br/>
<img src="https://github.com/audreycui/pix2para/blob/master/images/art_desc2455.jpg" height="220px" align="left">
<br/>"circles depict shore\\ a home white a through features warhol world the, a looking, state to,."

