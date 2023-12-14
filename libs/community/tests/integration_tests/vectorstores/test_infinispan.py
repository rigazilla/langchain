import json
from typing import List, Optional

from langchain_core.documents import Document
import requests
from langchain_community.vectorstores import Infinispan
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


# from tests.integration_tests.vectorstores.fake_embeddings import (
#    FakeEmbeddings,
#    fake_texts,
# )

def _infinispan_from_texts() -> Infinispan:
    return Infinispan.from_texts(embedding=FakeEmbeddings())


def test_infinispan_query_empty() -> None:
    infinispan = _infinispan_from_texts()
    output = infinispan.req_query("from Person")
    assert output.status_code == 200
    return

def test_infinispan_post() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_clear()
    output = infinispan.req_post("1", '{ "_type" : "Person", "name" : "Vittorio", "email" : "example@example.com"}')
    assert output.status_code == 204
    infinispan = _infinispan_from_texts()
    output = infinispan.req_post("1", '{ "_type" : "Person", "name" : "Vittorio", "email" : "example@example.com"}')
    assert output.status_code == 409
    return

def test_infinispan_post_then_query() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_clear()
    output = infinispan.req_post("1", '{ "_type" : "Person", "name" : "Vittorio", "email" : "milan@example.com"}')
    assert output.status_code == 204
    output = infinispan.req_post("2", '{ "_type" : "Person", "name" : "Tristan", "email" : "como@example.com"}')
    assert output.status_code == 204
    output = infinispan.req_post("3", '{ "_type" : "Person", "name" : "Fabio", "email" : "rome@example.com"}')
    assert output.status_code == 204
    output = infinispan.req_query("from Person")
    assert output.status_code == 200
    str = output.content.decode("utf-8")
    jOut = json.loads(str)
    assert jOut["hit_count"] == 3
    names = [x["hit"]["name"] for x in jOut["hits"]]
    names.sort()
    assert names == ["Fabio", "Tristan", "Vittorio"]

def test_infinispan_schema_post() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_clear()
    data = '''
/**
 * @Indexed
 */
message vector {
/**
 * @Vector(dimension=3)
 */
repeated float floatVector = 1;
string content = 2;
}
    '''
    output = infinispan.req_schema_post("vector.proto",data)
    assert output.status_code == 200

def test_infinispan_cache_delete() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_clear()
    output = infinispan.req_schema_delete("embeddingvectors")
    assert output.status_code == 204


def test_cache_post() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_clear()
    data = '''
{
  "distributed-cache": {
    "owners": "2",
    "mode": "SYNC",
    "statistics": true,
    "encoding": {
      "media-type": "application/x-protostream"
    },
    "indexing": {
      "enabled": true,
      "storage": "filesystem",
      "startup-mode": "AUTO",
      "indexing-mode": "AUTO",
      "indexed-entities": [
        "vector"
      ]
    }
  }
}
'''
    output = infinispan.req_cache_post("embeddingvectors",data)
    assert output.status_code == 200

def test_infinispan_schema_delete() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_clear()
    output = infinispan.req_schema_delete("vector.proto")
    assert output.status_code == 204

def test_infinispan_vectors_post_then_query() -> None:
    infinispan = _infinispan_from_texts()
    #infinispan.req_schema_delete("vector.proto")
    #test_infinispan_schema_post()
    infinispan.req_clear()
    populate(infinispan)
    output = infinispan.req_query("from vector i where i.floatVector <-> [7,7,7]~2")
    assert output.status_code == 200
    str = output.content.decode("utf-8")
    jOut = json.loads(str)
    assert jOut["hit_count"] == 2
    vectors = [x["hit"]["floatVector"] for x in jOut["hits"]]
    vectors.sort()
    assert vectors == [[3.0, 2.0, 3.0], [4.0, 2.0, 3.0]]

def test_infinispan_vectors_post_then_query_by_vector() -> None:
    infinispan = _infinispan_from_texts()
    infinispan.req_clear()
    populate(infinispan)
    output = infinispan.similarity_search_by_vector([7,7,7],2)
    assert output.status_code == 200
    str = output.content.decode("utf-8")
    jOut = json.loads(str)
    assert jOut["hit_count"] == 2
    vectors = [x["hit"]["floatVector"] for x in jOut["hits"]]
    vectors.sort()
    assert vectors == [[3.0, 2.0, 3.0], [4.0, 2.0, 3.0]]


def populate(infinispan) -> None:
    output = infinispan.req_post("1", '{"_type": "vector", "floatVector" : [ 1.0, 2.0, 3.0 ]}')
    output = infinispan.req_post("2", '{"_type": "vector", "floatVector" : [ 2.0, 2.0, 3.0 ]}')
    output = infinispan.req_post("3", '{"_type": "vector", "floatVector" : [ 3.0, 2.0, 3.0 ]}')
    output = infinispan.req_post("4", '{"_type": "vector", "floatVector" : [ 4.0, 2.0, 3.0 ]}')

# @pytest.mark.requires("sqlite-vss")
# def test_sqlitevss() -> None:
#     """Test end to end construction and search."""
#     docsearch = _sqlite_vss_from_texts()
#     output = docsearch.similarity_search("foo", k=1)
#     assert output == [Document(page_content="foo", metadata={})]


# @pytest.mark.requires("sqlite-vss")
# def test_sqlitevss_with_score() -> None:
#     """Test end to end construction and search with scores and IDs."""
#     texts = ["foo", "bar", "baz"]
#     metadatas = [{"page": i} for i in range(len(texts))]
#     docsearch = _sqlite_vss_from_texts(metadatas=metadatas)
#     output = docsearch.similarity_search_with_score("foo", k=3)
#     docs = [o[0] for o in output]
#     distances = [o[1] for o in output]
#     assert docs == [
#         Document(page_content="foo", metadata={"page": 0}),
#         Document(page_content="bar", metadata={"page": 1}),
#         Document(page_content="baz", metadata={"page": 2}),
#     ]
#     assert distances[0] < distances[1] < distances[2]


# @pytest.mark.requires("sqlite-vss")
# def test_sqlitevss_add_extra() -> None:
#     """Test end to end construction and MRR search."""
#     texts = ["foo", "bar", "baz"]
#     metadatas = [{"page": i} for i in range(len(texts))]
#     docsearch = _sqlite_vss_from_texts(metadatas=metadatas)
#     docsearch.add_texts(texts, metadatas)
#     output = docsearch.similarity_search("foo", k=10)
#     assert len(output) == 6
