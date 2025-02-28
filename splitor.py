import argparse
import concurrent.futures
import os
import time
from hashlib import md5

import docx
import filetype
import numpy as np
import torch
from loguru import logger
from transformers import BertModel, BertTokenizer


class Splitor:
    """A class for splitting files into chunks. Supports .txt, .docx formats.

    Attributes:
        dataPath (str): The path to the directory or file to be split.

    Optional Attributes:
        maxWorkers (int): The maximum number of threads for processing.
    """

    def __init__(
        self, dataPath: str, maxWorkers: int = 10, outputPath: str = "output.txt"
    ) -> None:
        self.__dataPath: str = dataPath
        self.__maxWorkers: int = maxWorkers
        self.__outputPath: str = outputPath
        self.__fileInfos: list[tuple[str, str]] = []
        self.__tempFilePaths: list[str] = []
        self.__model: BertModel = None
        self.__tokenizer: BertTokenizer = None
        self.__chunkCount: int = 0

    def run(self) -> None:
        """Starts the file processing workflow."""
        # 建立临时目录，存放缓存数据
        if not os.path.exists("temp"):
            os.mkdir("temp")

        # 覆盖输出文件
        open(self.__outputPath, "w+", encoding="utf-8").close()

        # 扫描目录文件
        self.__scanPath(self.__dataPath)
        if len(self.__fileInfos) > 0:
            self.__parseFiles()

        # 加载相似度模型
        self.__loadModel()

        # 文件切片
        self.__splitText()

        # 清理缓存文件
        self.__cleanTemp()

    def __scanPath(self, dataPath: str) -> None:
        """Recursively scans the given path and collects file paths.

        Args:
            dataPath (str): The path to scan. It can be a file or a directory.

        Raises:
            AssertionError: If the given path does not exist.
            Expection: If the path is neither a file nor a directory.
        """
        # 判断地址是否存在
        assert os.path.exists(dataPath), f"Path does not exist: {dataPath}"

        if os.path.isfile(dataPath):
            # 如果是文件
            self.__processFile(dataPath)
        elif os.path.isdir(dataPath):
            # 如果是目录，递归遍历目录下的所有文件
            for root, _, files in os.walk(dataPath):
                for file in files:
                    realPath = os.path.join(root, file)
                    self.__processFile(realPath)
        else:
            # 既不是文件也不是目录（这啥玩意儿）
            raise Exception(f"Invalid path: {dataPath}")

    def __processFile(self, filePath: str) -> None:
        """Processes a single file and adds it to the file paths list if valid.

        Args:
            filePath (str): The file path to process.
        """
        if filePath.endswith(".txt"):
            self.__fileInfos.append((filePath, "txt"))
            return

        try:
            kind = filetype.guess_extension(filePath)
        except TypeError:
            logger.warning(f"Unknown file type: {filePath}")
        else:
            if kind is None:
                logger.warning(f"Unknown file type: {filePath}")
            elif kind in ("txt", "docx"):
                self.__fileInfos.append((filePath, kind))
            else:
                logger.warning(f"Unsupported file type: {filePath} (Detected: {kind})")

    def __parseFiles(self) -> None:
        """Parse the collected files using multithreading."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.__maxWorkers
        ) as executor:
            futures = []
            for file in self.__fileInfos:
                futures.append(executor.submit(self.__parser, file))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Error while processing a file: {exc}")

    def __parser(self, fileInfo: tuple[str, str]) -> None:
        """Parse a file based on its type.

        Args:
            fileInfo (tuple[str, str]): A tuple containing the file path and file type.
        """
        filePath, fileType = fileInfo
        if fileType == "docx":
            self.__tempFilePaths.append(self.__docxParser(filePath))
        if fileType == "txt":
            self.__tempFilePaths.append(self.__txtParser(filePath))

    def __docxParser(self, filePath: str) -> str:
        """Parses a .docx file and processes its content.

        Args:
            filePath (str): Path to the .docx file.

        Returns:
            str: Path to the generated text file.

        Raises:
            Expection: Can not parse the docx file.
        """
        logger.info(f"Parsing docx file: {filePath}")

        tempPath = os.path.join(
            "temp",
            md5(f"{time.time()}-{filePath}".encode("utf-8")).hexdigest() + ".txt",
        )
        try:
            file = docx.Document(filePath)
        except Exception as e:
            raise Exception(f"Cannot parse DOCX file: {filePath}") from e
        else:
            with open(tempPath, "w+", encoding="utf-8") as wf:
                for p in file.paragraphs:
                    p = p.text.strip("\n").strip(" ")
                    if p == "":
                        continue
                    wf.write(p + "\n")

            return tempPath

    def __txtParser(self, filePath: str) -> str:
        """Parse a .txt file and processes its content.

        Args:
            filePath (str): Path to the .txt file.

        Returns:
            str: Path to the generated text file.
        """
        logger.info(f"Parsing txt file: {filePath}")

        tempPath = os.path.join(
            "temp",
            md5(f"{time.time()}-{filePath}".encode("utf-8")).hexdigest() + ".txt",
        )

        with open(tempPath, "w+", encoding="utf-8") as wf:
            f = open(filePath, "r", encoding="utf-8")
            for eachline in f.readlines():
                eachline = eachline.strip("\n").strip(" ")
                if eachline == "":
                    continue
                wf.write(eachline + "\n")

        return tempPath

    def __loadModel(self) -> None:
        """Load model from huggingface."""
        logger.info("Loading bert-base-chinese model...")

        self.__tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese", cache_dir="models"
        )
        self.__model = BertModel.from_pretrained(
            "bert-base-chinese", cache_dir="models"
        )

    def __splitText(self) -> None:
        """Split text via simimarity."""
        assert self.__tokenizer != None, "tokenizer is None"
        assert self.__model != None, "model is None"

        chunk = ""
        lastLine = ""
        context = ""
        chunkSize = 512  # 最小块大小
        overlapSize = 200  # 重叠部分大小
        similarityThreshold = 0.9  # 相似度阈值

        for eachPath in self.__tempFilePaths:
            with open(eachPath, "r", encoding="utf-8") as f:
                for eachline in f.readlines():
                    if not lastLine:
                        lastLine = eachline
                        chunk += eachline
                        continue

                    # 计算相似度
                    similarity = self.calc_simimarity(lastLine, eachline)

                    # 如果相似度高或者当前块长度不够512，则继续追加
                    if similarity >= similarityThreshold or len(chunk) < chunkSize:
                        chunk += eachline
                    else:
                        self.__saveChunk(chunk)
                        chunk = chunk[-overlapSize:] + eachline  # 重新开始新的块

                    lastLine = eachline

                # 处理文件最后的剩余部分
                if chunk:
                    self.__saveChunk(chunk)

        logger.success(f"splited {self.__chunkCount} chunks")

    def __saveChunk(self, chunk: str) -> None:
        """Save chunk to file.

        Args:
            chunk (str): Chunk content.
        """
        chunk = chunk.strip("\n").strip(" ")
        logger.info(f"\n<chunk>\n{chunk}</chunk>\nChunk Size: {len(chunk)}")
        with open(self.__outputPath, "a+", encoding="utf-8") as wf:
            wf.write(f"<chunk>\n{chunk}</chunk>\n")

        self.__chunkCount += 1

    def __cleanTemp(self) -> None:
        """Remove temp file."""
        logger.info("Cleaning temp file")

        for each in self.__tempFilePaths:
            os.remove(each)

    def calc_simimarity(self, s1: str, s2: str):
        """Calculate text simimarity.

        Args:
            s1 (str): Given text 1.
            s2 (str): Given text 2.
        """
        inputs = self.__tokenizer(
            [s1, s2], return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.__model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="file splitor")

    # 数据地址
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset, can be a directory or a file.",
    )
    # 最大线程数
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of worker threads for file processing.",
    )
    # 输出文件
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.txt",
        help="Path to the output file.",
    )

    args = parser.parse_args()

    splitor = Splitor(args.data_path, args.max_workers, args.output_path)
    splitor.run()
    splitor.run()
    splitor.run()
