# A marketing campaign management policy simulator.
This application simulates the implementation of candidate Policies in a Telecommunications company.
The application allowes multiple policies to run in paralel over the same customer base to provide the likely cumulative customer lifetime value per policy

![Drag Racing](test.png)

# Current policies
## EpsilonRingtail
This is a policy based on the Epsilon greedy methode.
but where instead of optimizing for minimal regret we optimize for maximum average customer lifetime value.

## BayesianGroundhog
This is a policy based on Thomson sampling from a Beta distribution of product convert rates.
And the sampled conversion rate is then multiples by the average reward to get teh expected customer lifetime value of the action.
The algorithm then chooses the action with teh maximum expected customer lifetime value.

## RandomCrayfish
This is simply a reference policy that pick a random action.

## Instalation
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

## Running the application
```bash
python simulator.py
```

## Running the dashboard
```bash
streamlit run app.py
```

## Making your own Policy
First create a new package folder in the root of the repository. 
Try and give it a cool name, see https://pypi.org/project/coolname/ to see how to generate one.

Now copy the __init__.py file from the randomCrayfish folder into you new folder as a starting point.
This is where you can implement you policy.
Take a look at the methods you can implement in the "Policy" class in the  "policy.py" file.

Onces your ready to test you can add your new package(policy) the "policies" list in the "simulator.py" file. This is the line right after the
```python
if __name__ == "__main__":
```