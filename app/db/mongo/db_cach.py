import redis
import mongoengine
import json
from mongoengine.queryset import QuerySet
from app.db.redis.redis import get_redis_cluster
from app.db.mongo.connect import connect_to_mongo
from typing import Optional
from redis.asyncio.cluster import RedisCluster

# --- 1. Connect to Databases ---
# Connect to a running Redis instance
# decode_responses=True is important so we get strings back from Redis, not bytes
global redis_client
redis_client: Optional[RedisCluster] = None


# --- 2. Create the Caching Logic in a Custom QuerySet ---
class CachingQuerySet(QuerySet):
    """
    A custom QuerySet that implements read-through caching for .get() calls.
    """
    # Set a default cache timeout of 5 minutes
    CACHE_TTL_SECONDS = 300

    async def get(self, *args, **kwargs):
        """
        Overrides the default .get() method to add caching.
        """
        # Only cache simple primary key lookups for now
        if len(args) == 0 and len(kwargs) == 1 and 'id' in kwargs:
            pk_value = kwargs['id']
            cache_key = f"{self._document._get_collection_name()}:{pk_value}"
            
            # --- Check the cache first ---
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                print(f"CACHE HIT for key: {cache_key}")
                # Re-create the MongoEngine document from the cached JSON data
                cached_json = json.loads(cached_data)
                # Convert the ObjectID back to a proper format
                if '_id' in cached_json and isinstance(cached_json['_id'], dict) and '$oid' in cached_json['_id']:
                    cached_json['_id'] = cached_json['_id']['$oid']
                doc = self._document._from_son(cached_json)
                return doc

            print(f"CACHE MISS for key: {cache_key}")

        # --- If it was a miss (or not a simple pk query), hit the DB ---
        doc = super().get(*args, **kwargs)
        
        # --- Cache the result (if it was a simple pk query) ---
        if len(args) == 0 and len(kwargs) == 1 and 'id' in kwargs:
            pk_value = kwargs['id']
            cache_key = f"{self._document._get_collection_name()}:{pk_value}"
            
            # Convert the document to JSON and cache it
            doc_json = doc.to_json()
            await redis_client.setex(cache_key, self.CACHE_TTL_SECONDS, doc_json)
            print(f"CACHED document with key: {cache_key}")
        
        return doc

    def delete(self, *args, **kwargs):
        """
        Overrides delete to ensure cache invalidation.
        This is a simple implementation that assumes you first .get() the object
        and then call .delete() on it.
        """
        import asyncio
        
        # Helper function to handle async cache invalidation
        async def _invalidate_cache():
            for doc in self:
                cache_key = f"{doc._get_collection_name()}:{str(doc.id)}"
                await redis_client.delete(cache_key)
                print(f"CACHE DELETE for key: {cache_key}")
        
        # Schedule the async cache invalidation without awaiting
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Schedule the cache invalidation task
            asyncio.create_task(_invalidate_cache())
        except RuntimeError:
            # No event loop running, skip cache invalidation
            print("No event loop running, skipping cache invalidation")
        
        # Call the original delete method
        return super().delete(*args, **kwargs)


# --- 3. DocumentCache Base Class ---
class DocumentCache(mongoengine.Document):
    """
    Base class that provides caching functionality for MongoEngine documents.
    Any document that inherits from this class will automatically get caching behavior.
    """
    
    meta = {
        'abstract': True,  # This makes it an abstract base class
        'queryset_class': CachingQuerySet
    }

    async def save(self, *args, **kwargs):
        """
        Overrides save to ensure cache invalidation.
        """
        # If this document already has an ID, invalidate its cache
        if self.id:
            cache_key = f"{self._get_collection_name()}:{str(self.id)}"
            await redis_client.delete(cache_key)
            print(f"CACHE DELETE for key: {cache_key}")
        
        # Call the original save method
        return super().save(*args, **kwargs)
    
    def delete(self, *args, **kwargs):
        """
        Overrides delete to ensure cache invalidation for single document deletion.
        """
        import asyncio
        
        # Helper function to handle async cache invalidation
        async def _invalidate_cache():
            if self.id:
                cache_key = f"{self._get_collection_name()}:{str(self.id)}"
                await redis_client.delete(cache_key)
                print(f"CACHE DELETE for key: {cache_key}")
        
        # Schedule the async cache invalidation and wait for it
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Create and run the task, but don't await it directly
            # Instead, schedule it and give it a moment to execute
            task = asyncio.create_task(_invalidate_cache())
        except RuntimeError:
            # No event loop running, skip cache invalidation
            print("No event loop running, skipping cache invalidation")
        
        # Call the original delete method
        return super().delete(*args, **kwargs)


# --- 4. See it in Action ---
if __name__ == '__main__':
    # --- 5. Book Document inheriting from DocumentCache ---
    class Book(DocumentCache):
        """
        Book document that inherits caching functionality from DocumentCache.
        """
        title = mongoengine.StringField(required=True)
        author = mongoengine.StringField()
        publication_year = mongoengine.IntField()

        meta = {
            'collection': 'books'
            # queryset_class is inherited from DocumentCache
        }

    async def main():
        # Connect to MongoDB
        await connect_to_mongo()

        # Connect to Redis
        global redis_client
        redis_client = await get_redis_cluster()

        # Clean up previous runs if there are books in mongo
        if Book.objects.count() > 0:
            Book.objects.all().delete()
            await redis_client.flushdb()
            print("--- Cleared DB and Cache ---\n")

        await asyncio.sleep(0.1) # wait 100ms

        # Create a new book
        print("--- Creating a new book ---")
        book = Book(title="The Hitchhiker's Guide to the Galaxy", author="Douglas Adams", publication_year=1979)
        await book.save() # This will trigger the cache invalidation in our custom save()
        book_id = book.id
        print(f"Book created with ID: {book_id}\n")

        await asyncio.sleep(0.1) # wait 100ms

        # --- First Read: Cache Miss ---
        print("--- First read attempt ---")
        retrieved_book = await Book.objects.get(id=book_id)
        print(f"Retrieved: {retrieved_book.title}\n")

        # --- Second Read: Cache Hit ---
        print("--- Second read attempt ---")
        retrieved_book_from_cache = await Book.objects.get(id=book_id)
        print(f"Retrieved: {retrieved_book_from_cache.title}\n")

        await asyncio.sleep(0.1) # wait 100ms

        # --- Update the book ---
        print("--- Updating the book ---")
        retrieved_book_from_cache.publication_year = 1980
        await retrieved_book_from_cache.save() # This will trigger the cache invalidation
        print("Book updated.\n")

        await asyncio.sleep(0.1) # wait 100ms   

        # --- Read after update: Cache Miss again ---
        print("--- Reading after update ---")
        updated_book = await Book.objects.get(id=book_id)
        print(f"Retrieved updated year: {updated_book.publication_year}\n")

        await asyncio.sleep(0.1) # wait 100ms

        # --- Read again: Cache Hit again ---
        print("--- Reading again after update ---")
        updated_book_from_cache = await Book.objects.get(id=book_id)
        print(f"Retrieved updated year: {updated_book_from_cache.publication_year}\n")

        # --- Delete the book ---
        print("--- Deleting the book ---")
        updated_book_from_cache.delete() # This will trigger cache invalidation
        # Give the async cache invalidation task time to complete
        await asyncio.sleep(0.1) # wait 100ms
        print("Book deleted.\n")

        # --- Verify it's gone from cache ---
        cache_key = f"books:{book_id}"
        final_cache_check = await redis_client.get(cache_key)
        print(f"Final check for key {cache_key} in cache: {final_cache_check}")
    
    import asyncio
    asyncio.run(main())
