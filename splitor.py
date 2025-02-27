import time
import os
import sys
import docx
import argparse
import concurrent.futures
import filetype
from hashlib import md5
from loguru import logger


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

    def run(self):
        """Starts the file processing workflow."""
        # 建立临时目录，存放缓存数据
        if not os.path.exists("temp"):
            os.mkdir("temp")

        self.__scanPath(self.__dataPath)
        if len(self.__fileInfos) > 0:
            self.__parseFiles()
        
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
                wf.write(eachline+"\n")

        return tempPath

    def __cleanTemp(self):
        """Remove temp file.
        """
        logger.info("Cleaning temp file")
        for each in self.__tempFilePaths:
            os.remove(each)


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
