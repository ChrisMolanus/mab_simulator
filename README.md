# A marketing campaign management policy simulator.
This application simulates the implementation of candidate Policies in a Telecommunications company.
The application allows multiple policies to run in parallel over the same customer base to provide the likely 
cumulative delta customer lifetime value per policy

![Policy revenue timeline](images/test.png)

# Current policies
The current policies implemented are examples of popular algorithms applied to Multi-Arms Bandit problems. 
This is my no means an exhaustive list, and the implemented policies are not the standard methode you would find on [Wikipedia](https://en.wikipedia.org/wiki/Multi-armed_bandit). 
Typical implementations you can find online focus on optimizing conversion rate. 
The implementations here focus on optimizing Delta Customer Lifetime Value

##SegmentJunglefowl
This is an implementation that simulates the standard Marketing VIP Gold Silver Bronze segmentation. For simplicity, 
we only implemented a three tier marketing segmentation
It is intended to be seen as reference for how a marketing department might work 
if the customer segments and actions where mapped by hand.

![SegmentJunglefowl timeline](images/SegmentJunglefowl.png)

## EpsilonRingtail
This is a policy based on the Epsilon greedy methode, which take a Semi-Uniform strategy. 
Meaning the most profitable campaign for this customer is allocated to the customer 
except for when a random campaign is taken.
This implementation optimize for maximum average customer lifetime value(profit) instead of minimizing regret.
Taking a random campaign every so now and then attempts to give what may seem like sub-optimal campaigns at the time a chance.

![EpsilonRingtail timeline](images/EpsilonRingtail.png)

## BayesianGroundhog
This is a policy based on Thomson sampling from a Beta distribution of product convert rates.
And the sampled conversion rate is then multiples by the average customer lifetime value(reward)
to get the Expected customer lifetime value of the campaign(action).
The algorithm then chooses the action with the maximum expected customer lifetime value.

![BayesianGroundhog timeline](images/BayesianGroundhog.png)

## RandomCrayfish
This is simply a reference policy that pick a random action.

![RandomCrayfish timeline](images/RandomCrayfish.png)

# Installation
Clone the repository to a local directory

Change the current working directory to in the folder where created when the repository was cloned

Then create a virtual environment
```bash
python -m venv venv
```
Then activate the environment
```bash
source venv\bin\activate
```
Install all the necessary packages
```bash
pip install -r requierments.txt
```

# Running the application
```bash
python simulator.py
```

# Running the dashboard
```bash
streamlit run app.py
```
![Dashboard screen shot](images/dashboard_screenshot.png)

# Making your own Policy
First create a new package folder in the root of the repository. 
Try and give it a cool name, see https://pypi.org/project/coolname/ to see how to generate one.

Now copy the \_\_init\_\_.py file from the randomCrayfish folder into you new folder as a starting point.
This is where you can implement your policy.
Take a look at the methods you can implement in the "Policy" class in the  "policy.py" file.

The policy is designed to function as a (near)real-time transformer. This means that it is expected 
to be capable for producing an NBA at any moment.

![Start Trigger](images/start_trigger.svg)

Customer events such as a call to the service desk, or a visit to the website trigger an evaluation for 
a new Next Best Action(NBA). For simplicity, we consider the business ruling that limits the possible things 
we are allowed to do with this customer (Actions) as Out-Of-Scope. These business rules could be 
for example: 
- We can not sell a 1 Gigabit Internet connection to the customer 
because the cable going to their house can never provide that bandwidth.
- We can only sell an upgrade connection to the customer and not a second line 
  since their address can not have a second line installed

Handling changes to these rules are considered out of scope, here we are only concerned with 
learning from past actions to better suggest new NBAs. To accomplish this your policy must implement its own 
calibration(or training) cycle. The choice of how often to recalibrate is up to you. 
This could be:
- Once we have more than 100 new samples (of action rewards)
- Once a day
- On every new sample that we get since we get them in real-time one by one.

There is no computational limit set here, so the only consideration is 
to decide when and how often you should update your belief (model). 
Here it is important to note that when a new action is introduced it is not known if it is a bad one or a good one, 
whether it will have a high reward, or a negative reward. 
So the longer we wait to evaluate the action's probability of producing a positive reward 
the more your policy may prescribe it as an NBA that can result in larger losses if it is a bad one. 
The Action doesn't even have to produce a negative reward on average to be a bad NBA, 
it just has to be worse that any other action that you could have prescribed 
that might have had more of a chance of producing a higher reward. This is what is called regret, the difference between
the average reward of an action, and the estimated average reward of the best performing action 
that could have been prescribed to this customer at teh time.

Once your policy provides an NBA for the customer it is places on a queue 
for it to be executed by the channel associated with the Action.
Note that the action is not performed immediately, in the case of an outbound call for example 
it can be hours before this cll can be made by an agent.

![execute trigger](images/execute_trigger.svg)

Once the action has been taken, a flag is set in the customer's profile that the action has been take and no other actions should be taken until ether:
- The customer performs an action
- The cool off period has pasted, and the action is presumed to have had no effect.

Both these scenarios produce a (pseudo) customer action. If the customer actually did something we can measure
it is simple, the event is what ever the customer did or at least the data that we measured. 
If the customer did nothing, and the cool off period has pasted 
the company generated a "NULL action" indicating that the company has given up on waiting to see 
if the action will have an effect. 

![reward trigger](images/reward_trigger.svg)

Each Action has its own cool-off period length based on the marketeers estimates, 
The length of the cool-off period is also considered out of scope here. These reward events can be used 
by your policy to calibrate the presumed effect for the NBA that was previously performed on the customer.


You can test your new package(policy) by adding it to the "policies" list in the "simulator.py" file. 
This is the line right after the
```python
if __name__ == "__main__":
```

## Class diagram
![Class diagram](images/class_diagram.svg)
