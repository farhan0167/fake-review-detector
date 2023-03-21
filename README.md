# Fake Review Application

This is an extension of my class project on [Generating and Detecting Fake Reviews](https://github.com/farhan0167/AIGeneratedFakeReview). The purpose behind this implementation was
to demo our work by letter users generate fake reviews based on the models we fine-tuned. Another purpose to do this was to get hands-on experience working with Docker, Docker compose, NGINX, 
AWS EC2 and ML deployment following my course in Data Centers and Cloud Computing.

Since this was originally not intended, we have been training/fine-tuning our models in Colab and storing our model weights in Google Drive, which quickly became a challenge. Here are some of them:
1. Since model weights were stored in the Drive, downloading 1.5GB of model weights were taking forever.
2. A lot of the fine-tuning implementation was done using Colab servers with GPU, resources that I did not have on my local machine. Even running a jupyter notebook of the colab files was crashing my computer.

As a result of this, I needed to move to the cloud again and my choice was AWS EC2 instance. I created an EC2 instance of type t2.medium. Intially I chose a t2.nano but running an inference was taking more than 2 minutes so I chose t2.medium which returns a response in 30s (not ideal but was adequate for a demo). The first challenge, however, was to get one of the model weights
on to EC2. Since downloading on localhost and then re-uploading did not make sense, I moved the weights on to AWS S3 and then on the EC2 instance, I mounted the weights from the S3 bucket. This approach was much faster than waiting for the weights to download on my computer. 

The application is simple. The main code resides [here](https://github.com/farhan0167/fake-review-detector/blob/main/backend/app.py#L56:~:text=%5D)). The endpoint takes in the prompt and the maximum number of words you want to generate, and returns you a generated review. 

The Dockerfile can be found [here](https://github.com/farhan0167/fake-review-detector/blob/main/backend/Dockerfile) \
The Dockercompose file [here](https://github.com/farhan0167/fake-review-detector/blob/main/docker-compose.yml) \
The Nginx configuration files [here](https://github.com/farhan0167/fake-review-detector/blob/main/nginx/nginx.conf)

Something interesting that I learnt deploying the application on EC2 was running an application on a cloud instance with an IP address will always give you a connection over http. In order to make the connection go over https we will need a certificate from a certificate authority. We can certainly use our own certificate but there is nothing to say that Bob next door couldnt open up another certificate with my name claiming to be me- how do you verify that Bob is not me, and that I am actually me who I claim to be? That's where certificate authorities come in- its an organization that the internet community has agreed to trust. They sign our certificates validating I am who I claim to be. But up until that point I thought this was a paid service but behold- Lets Encrypt lets gives you a CA certificate by running a simple command. You can learn more about how you can get CA certified certificates from [here](https://certbot.eff.org/).

