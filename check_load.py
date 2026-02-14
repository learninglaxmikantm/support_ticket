import joblib

p = joblib.load("priority.joblib")
c = joblib.load("category.joblib")

print(p.predict(["VPN not working"]))
print(c.predict(["Email login issue"]))