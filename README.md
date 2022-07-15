# display_dtree_webapp
A web app that allows users to upload a csv and then see a rendered decision tree of their data.

App can be accessed here:\
[CreateDecisionTree](https://joseph-foley-display-dtree-webapp-createdecisiontree-bac-6w6lye.streamlitapp.com/ "Title")

Click "How does this work" button for usage instructions.

Typical result:

![](Images/class_tree.png?raw=true)

Alternatively you can run this app locally by cloning this repo and installing the requirements (see requirements.txt)

A docker file is also present if you wish to run the app in a docker container.
##

I made this app so that others could easily make and visualise decision trees of their data without needing to know how to use R or Python.

I also made this app so that I could get hands on with some new technologies that interested me.
I'll briefly explain how I put this together using these technologies.

##

**Backend code (dtree_modelling.py)**

Pandas data handling. Anticipation of various issues that may come with datasets.

Simple sklearn dtree model.

Create an image object of the tree using graphviz, pydot, PIL and io.

##

**dtree_string.py**

Utility script for customizing the trees so that they are more distinguishable from the defaults.

Features:

Colour changes by converting hex to rgb and back again.

String manipulation of dtree using regular expressions.

##

**Frontend code (createdecisiontree.py)**

Streamlit app

The actual html template and favicon are edited. The script has to copy them from the repo and then dump them into the actual streamlit library when it runs.

##

**Docker: **Dockerised the app so that it could be easily deployed on AWS.

**AWS Elastic Beanstalk: **Works well with Docker to keep deployment simple.

**AWS Route 53: **Allowed me to have a custom domain name for the app (CreateDecisionTree.com)

**AWS Certificate Manager: **Secure the app with https (requires load balancing)

**Google Analytics: **Embedded trackers into the app so that I could keep an eye on traffic.

More info on the tech used here: [making_of_app](https://github.com/Joseph-Foley/display_dtree_webapp/blob/main/Docs/Making_of_app.md "Title")

Please note that after a few months I pulled the app off of AWS to save on costs. The app is now on the streamlit cloud service (see link above).
##

Finally, I also wrote a medium article that explains to the layperson how they can use decision trees generated by the app for classification and regression problems.

[medium article](https://medium.com/mlearning-ai/a-neat-web-app-to-map-out-your-data-with-a-decision-tree-b9f77c5600c7 "Title")