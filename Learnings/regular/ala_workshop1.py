import uuid

class UpiTxn:
    # What are the arguments for construction?
    def __init__(self, sender, reciever, amount):
        # create a uuid instance
        self.txn_id = uuid.uuid4
        self.sender = sender
        self.receiver = reciever
        self.amount = amount
        pass

class UPIPaymentTxn(UpiTxn):
    # What are the arguments for construction?
    def __init__(self, sender, receiver, amount):
        # pass the responsibility to 'UpiTxn'
        pass

class UPIReceiptTxn(UpiTxn):
    # What are the arguments for construction?
    def __init__(self, sender, reciever, amount):
        # pass the responsibility to 'UpiTxn'
        pass


class UpiTxnResponse:
    def __init__(self):
        pass