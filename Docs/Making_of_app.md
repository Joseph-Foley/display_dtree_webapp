**Backend code (dtree_modelling.py)**

Pandas data handling. Anticipation of various issues that may come with datasets.

Simple sklearn dtree model.

Create an image object of the tree using graphviz, pydot, PIL and io.  
## 

**dtree_string.py**

Utility script for customizing the trees so that they are more distinguishable from the defaults

Features:

Colour changes by converting hex to rgb and back again.

String manipulation of dtree using regular expressions.
## 

**Frontend code (createdecisiontree.py)**

Streamlit app

The actual html template and favicon are edited. The script has to copy them from the repo and then dump them into the actual streamlit library every time it runs.

An extra argument is needed to change the file upload limit when running the app.
## 

**Docker**

<https://towardsdatascience.com/how-to-deploy-a-semantic-search-engine-with-streamlit-and-docker-on-aws-elastic-beanstalk-42ddce0422f3>

^very good guide that I did not deviate from (later change file paths to what I wanted once I was comfortable with it).

Docker engine needs to be running before you can do everything that's needed. Just start the program on windows.

Beware that you're entering linux (Debian) territory now and the app was less forgiving when it cam to capitalization of file extensions (e.g. PNG & png).

Docker file was simple business. Didn't take much research to include Graphviz in there as well (though I cannot yet specify the version I want as of writing).

3 key command line commands

    #build the image
    
    docker build -t josephfoley/dtree_app .
    
    #run a container of the image on local machine (to see if it actually works or not)
    
    docker run -p 8501:8501 josephfoley/dtree_app
    
    #Push to Docker Hub repo
    
    docker push josephfoley/dtree_app

The repo on Dockerhub is public because it makes working with AWS easier. CBA to figure out how to do it private. My work isn't as exposed there as it is on github so no big deal. See link above for Docker Hub stuff.

Also worth mentioning that if you have a Docker Container running locally, you can tunnel (SSH?) into it. You'd do this to check that files and directories went to where you want them to be.

## 
**AWS Elastic Beanstalk**

Link in Docker segment explains most of what to do.

By default they limit user uploads to my app by 1mb. I found a work around but its not ideal as I think it's a solution for just once instance -- it may not transfer to new instances when they spin up to support the load. Maybe it will work, dunno.

## 
The fix:


select your instance in the EC@ dash:

https://us-east-2.console.aws.amazon.com/ec2/v2/home?region=us-east-2#Instances:

connect to the instance (a terminal in browser will appear).

enter these commands


    cd ..
    sudo vim etc/nginx/nginx.conf
    
    #input (above access_log line):
    client_max_body_size 5M;

    #press esc
    #enter these commands
    :w
    :q
    sudo systemctl reload nginx
    exit
## 

**AWS Route 53 (Custom Domain Name for App)**

<https://towardsdatascience.com/build-a-serverless-website-using-amazon-s3-and-route-53-c741fae6ef8d>

^good guide

You need to actually added the prefix "www."  Yourself when making records (have one with and one without).

You get http only, no https and so no padlock on chrome. There will be a warning that the site is not secure.

## 
**AWS Certificate Manager (SSL / HTTPS / Secure Certification)**

<https://medium.datadriveninvestor.com/configurate-route-53-and-adding-ssl-certificate-145d8a317d91>

You can link certificate to Route 53 domain from earlier.

However, one has to enable load balancing on their Elastic Beanstalk app in order to apply the certificate. ([Configuring Your Elastic Beanstalk Environment's Load Balancer to Terminate HTTPS](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/configuring-https-elb.html).)

This is how you enable load balancing:

<https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/using-features-managing-env-types.html#using-features.managing.changetype>

**But!!** You must switch the **listener protocol** from https to TCP (streamlit has issues loading the full app otherwise). <https://stackoverflow.com/questions/27436110/websockets-with-aws-and-elastic-beanstalk/39604687>

**And then!** You just gotta physically type "https" along with your URL in web browser for this to work. So yes, the unsecure http version of the site is still there. Just never go there once this is done.

Note: Enabling Load balancing costs 50cents a day (slightly more than a T3 small instance).
## 
**Google Analytics**

Need this to see page views of the web app (could not figure out how to do that with AWS).

<https://discuss.streamlit.io/t/how-to-add-google-analytics-or-js-code-in-a-streamlit-app/1610/10>

^Suryate's post.

As discussed in the front end segment, there is a default html (index.html) file in the streamlit directory that is used for the app. To edit this, I had a copy of that file in my repo that contained the GA javascript code in the HEAD (cannot edit the HEAD using the streamlit library directly). When the app runs, this file is copied and then overwrites the file in the streamlit library directory. It has to be done this way because I cant manually move it over if Im deploying with Docker (I suspect it may also be possible to do this in the Docker file itself but I don't know how). The custom favicon (little icon that you see in the tabs of your web browser) is also handled I the same way.

## 
**SEO**

Jack says there is very little one can do on their own site for this. Having a link of your site on other sites is very powerful.

I mostly just added meta tags in the head (placed under GA js).



    <meta name="description" content="Free Online ML Decision Tree Creator. Just drag & drop your data!">
    
    <meta name="keywords" content="Machine Learning, Decision Tree, Classification, Regression, Free, Online">
    
    <meta property="og:title" content="Machine Learning Decision Tree Creator"/>
    
    <meta property="og:description" content="Free Online ML Decision Tree Creator. Just drag & drop your data!"/>
    
    <meta property="og:image" content="http://createdecisiontree.com/media/8f84fad9fd5f0e8deaec041f3d689c2ec83aa8469b21d8a8ad5de50d.png"/>
    
    <meta name="twitter:title" content="Machine Learning Decision Tree Creator">
    
    <meta name="twitter:description" content="Free Online ML Decision Tree Creator. Just drag & drop your data!">
    
    <meta name="twitter:image" content="http://createdecisiontree.com/media/8f84fad9fd5f0e8deaec041f3d689c2ec83aa8469b21d8a8ad5de50d.png">
    
    <meta name="twitter:card" content="summary_large_image">


<https://search.google.com/search-console?resource_id=http%3A%2F%2Fcreatedecisiontree.com%2F>

^this lets you know if Google can see your site (and list it in search results).

Initially it could not but you can request indexing there and then it will.
## 

**Medium**

Signed up. Wrote in Word. Pasted in their editor (you write "stories"). Simple as.

Admins of Medium partners contacted me to publish on their segment. I obliged:

<https://medium.com/mlearning-ai>

<https://medium.com/mlearning-ai/a-neat-web-app-to-map-out-your-data-with-a-decision-tree-b9f77c5600c7>