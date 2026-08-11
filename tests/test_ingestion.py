import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.chunker import split_documents
from src.loader import load_documents
from src.manifest import (
    build_manifest,
    corpus_description,
    index_is_current,
)
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    publish_vector_store,
    save_vector_store,
    validate_saved_index,
)


class DeterministicEmbeddings(Embeddings):
    """Petit modèle local réservé aux tests, sans téléchargement réseau."""

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


class LoaderTests(unittest.TestCase):
    def test_loads_supported_files_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "b.txt").write_text("Deuxième document", encoding="utf-8")
            (folder / "a.txt").write_text("Premier document", encoding="utf-8")
            (folder / "ignore.csv").write_text("ignoré", encoding="utf-8")

            documents = load_documents(folder)

        self.assertEqual(
            [document.metadata["source_name"] for document in documents],
            ["a.txt", "b.txt"],
        )

    def test_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Aucun document exploitable"):
                load_documents(directory)

    def test_rejects_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            load_documents("dossier-qui-n-existe-pas")


class ChunkerTests(unittest.TestCase):
    def test_splits_documents_and_keeps_metadata(self):
        document = Document(
            page_content="Un texte suffisamment long pour être découpé en passages.",
            metadata={"source_name": "source.txt"},
        )

        chunks = split_documents([document], chunk_size=25, chunk_overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk.metadata["source_name"] == "source.txt" for chunk in chunks)
        )

    def test_rejects_invalid_parameters(self):
        document = Document(page_content="Texte")

        with self.assertRaises(ValueError):
            split_documents([document], chunk_size=0, chunk_overlap=0)
        with self.assertRaises(ValueError):
            split_documents([document], chunk_size=10, chunk_overlap=10)
        with self.assertRaises(ValueError):
            split_documents([], chunk_size=10, chunk_overlap=0)


class VectorStoreTests(unittest.TestCase):
    def test_creates_and_saves_one_vector_per_chunk(self):
        chunks = [
            Document(page_content="Premier passage", metadata={"source_name": "a.txt"}),
            Document(page_content="Deuxième passage", metadata={"source_name": "b.txt"}),
        ]
        vector_store = create_vector_store(chunks, DeterministicEmbeddings())

        self.assertEqual(vector_store.index.ntotal, len(chunks))

        with tempfile.TemporaryDirectory() as directory:
            save_vector_store(vector_store, directory)
            output = Path(directory)
            self.assertTrue((output / "index.faiss").is_file())
            self.assertTrue((output / "index.pkl").is_file())

    def test_atomically_publishes_a_validated_index_and_manifest(self):
        chunks = [Document(page_content="Passage", metadata={"source_name": "a.txt"})]
        vector_store = create_vector_store(chunks, DeterministicEmbeddings())
        manifest = {
            "signature": "test",
            "results": {"pages": 1, "chunks": 1},
        }

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "faiss_index"
            publish_vector_store(vector_store, target, manifest)
            validate_saved_index(target, expected_vectors=1)
            saved_manifest = json.loads(
                (target / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(saved_manifest["signature"], "test")

    def test_loads_and_searches_saved_index(self):
        embeddings = DeterministicEmbeddings()
        chunks = [
            Document(
                page_content="Le projet analyse des indicateurs.",
                metadata={"source_name": "rapport.pdf", "page": 2},
            )
        ]
        vector_store = create_vector_store(chunks, embeddings)

        with tempfile.TemporaryDirectory() as directory:
            save_vector_store(vector_store, directory)
            loaded = load_vector_store(directory, embeddings)
            results = loaded.similarity_search("indicateurs", k=1)

        self.assertEqual(results[0].metadata["source_name"], "rapport.pdf")


class ManifestTests(unittest.TestCase):
    def test_detects_an_unchanged_then_modified_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            index = root / "index"
            documents.mkdir()
            source = documents / "source.txt"
            source.write_text("version 1", encoding="utf-8")
            files = corpus_description(documents)
            manifest = build_manifest(files, 500, 100, page_count=1, chunk_count=1)
            vector_store = create_vector_store(
                [Document(page_content="version 1")], DeterministicEmbeddings()
            )
            save_vector_store(vector_store, index)
            (index / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            self.assertTrue(index_is_current(index, files, 500, 100))

            source.write_text("version 2", encoding="utf-8")
            modified_files = corpus_description(documents)
            self.assertFalse(index_is_current(index, modified_files, 500, 100))

    def test_rejects_a_manifest_without_index_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            index = root / "index"
            documents.mkdir()
            (documents / "source.txt").write_text("contenu", encoding="utf-8")
            files = corpus_description(documents)
            manifest = build_manifest(files, 500, 100, page_count=1, chunk_count=1)
            index.mkdir()
            (index / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            self.assertFalse(index_is_current(index, files, 500, 100))


if __name__ == "__main__":
    unittest.main()
