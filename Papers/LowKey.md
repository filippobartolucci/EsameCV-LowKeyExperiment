# LowKey Levereging Adversarial Attacks to Protect Social Media Users from Facial Recognition
In this paper, the authors propose a method to protect individual users from mass survellaince systems by enabling the users to add adversarial filters to thei images before sharing them online.

# Why
Companies like Clearview [1] scraped billions of identities from various social media and built a massive surveillance system. Now anybody can take a person’s photo on the street and just find out their identity. So to protect your identity, you can not share your photos online, or you can ‘cloak’ your images before sharing them online. (Disclaimer: The paper does not claim to protect against Clearview systems as of now.)

# How 
![LowKey Attack](https://miro.medium.com/v2/resize%253Afit%253A1400/format%253Awebp/1%252A9NjVcBRzJbnHMAW-zGnthQ.png)

Gallery images are the ones in person’s control, like the images you share on social media. And probe image is the photo taken without your permission (say when you are jogging in a park, and someone captures a photo of you). LowKey suggests user to upload modified images online to poison the facial recognition model. The main idea is, the modified image should look as similar to the original image, but in the feature space, both the images should be as far as possible.

!LowKey Optimization](https://miro.medium.com/v2/resize%253Afit%253A1400/format%253Awebp/1%252AAtuddZd205YW84cvduFwJA.png)