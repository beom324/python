from pymongo import MongoClient

    #Provide the mongodb atlas url to connect python to mongodb using pymongo
CONNECTION_STRING = "mongodb://localhost:27017"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
db=client.sist2
collection=db["member"]

def listMember():
    return collection.find()

def insertMember(id,name,age):
    collection.insert_one({
        "id":id,
        "name":name,
        "age":age
    })
def insertOne(doc):
    return collection.insert_one(doc).inserted_id

# This is added so that many files can reuse the function get_database()
