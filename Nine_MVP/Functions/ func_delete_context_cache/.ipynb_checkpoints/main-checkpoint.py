import functions_framework
import vertexai
from vertexai.preview import caching

@functions_framework.http
def delete_context_cache(request):
    """
        delete context cache
             
    """   
    for cached_context in caching.CachedContent.list():
        cache_id=cached_context.name.split("/")[-1]
        cached_content = caching.CachedContent(cached_content_name=cache_id)
        cached_content.delete()
  
    return {'status':'SUCCESS'}