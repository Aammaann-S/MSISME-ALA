import uuid
import random

def success_95percent() -> bool:
    return random.random() < 0.95

class UpiTxn:
    # What are the arguments for construction?
    def __init__(self, sender, reciever, amount):
        # create a uuid instance
        self.txn_id = uuid.uuid4()
        self.sender = sender
        self.receiver = reciever
        self.amount = amount
        # pass

class UPIPaymentTxn(UpiTxn):
    # What are the arguments for construction?
    def __init__(self, sender, receiver, amount):
        # pass the responsibility to 'UpiTxn'
        # pass
        super().__init__(sender, receiver, amount)

class UPIReceiptTxn(UpiTxn):
    # What are the arguments for construction?
    def __init__(self, sender, receiver, amount):
        # pass the responsibility to 'UpiTxn'
        # pass
        super().__init__(sender, receiver, amount)


# class UpiTxnResponse: 
#     def __init__(self):
#         pass

class UpiTxnResponse:

    def __init__(self, transaction):
        self.txn_id = transaction.txn_id

        if success_95percent():
            self.status = "SUCCESS"
        else:
            self.status = "FAILED"


N = 1000
success_count = 0
failure_count = 0

for i in range(N):

    payment = UPIPaymentTxn(
        "Aman@upi",
        "Shop@upi",
        10
    )

    response = UpiTxnResponse(payment)

    if response.status == "SUCCESS":
        success_count += 1
    else:
        failure_count += 1


success_rate = (success_count / N) * 100
failure_rate = (failure_count / N) * 100

print("Total transactions:", N)
print("Successful Count:", success_count)
print("Failed Count:", failure_count)
print("----------------------------------------")
print("Success rate:", round(success_rate,2),"%")
print("Failure rate:", round(failure_rate,2),"%")
print("----------------------------------------")