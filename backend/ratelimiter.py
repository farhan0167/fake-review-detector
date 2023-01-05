import time
#check how much time elapsed since last logged
t = time.localtime()

lastcheck = {
  'month': 1,
  'day': 4,
  'hour': 22,
  'min': 7
}
limitperhour = 1
print(lastcheck)

if t.tm_mon - lastcheck['month'] >1:
  print("connectionT reset")
else:
  if t.tm_mday - lastcheck['day'] >0:
    print("connectionT reset")
  else:
    print("still active-2")
    if t.tm_hour - lastcheck['hour'] >=limitperhour:
      print(f"its been {t.tm_hour - lastcheck['hour']} hrs since last checked")
      print("connectionT reset")
    else:
      print(f"its been {t.tm_min - lastcheck['min']} mins since last checked")
      