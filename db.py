import re

from dotenv import load_dotenv
import io
import os

import numpy as np
from storage3.utils import StorageException
from supabase import create_client, Client
from PyPDF2 import PdfReader

PAST_ASSIGNMENTS_BUCKET = 'ava-prod-past-assignments'
CURRENT_ASSIGNMENTS_BUCKET = 'ava-prod-assignments'

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
secret_key: str = os.getenv("SUPABASE_SECRET_KEY")

supabase_sec: Client = create_client(url, secret_key)
assert supabase_sec is not None


class DB:

    @staticmethod
    def upload_cached_file(user, filename, file):
        username = user.email.split('@')[0]
        path = DB.construct_path_past(username, filename)
        try:
            supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).remove(path)
        except StorageException:
            print("Could not delete cached file", path)
            pass

        try:
            supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).upload(path, file)
        except StorageException:
            print("Could not upload cached file", path)
            pass
        return True

    @staticmethod
    def read_past_file(user, filename):

        # check if the file is .npy or .txt
        if filename[-4:] == '.npy':
            return DB.__download_past_npy_file(user.email, filename)
        elif filename[-4:] == '.txt':
            return DB.__download_past_txt_file(user.email, filename)
        else:
            return DB.__download_past_bytes_file(user.email, filename)

    @staticmethod
    def __download_past_bytes_file(user_email, filename):
        try:
            username = user_email.split('@')[0]

            path = DB.construct_path_past(username, filename)
            downloaded = supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).download(path)

            bytesIO = io.BytesIO(downloaded)
            val = bytesIO.getvalue()

            if type(val) == bytes:

                if downloaded.startswith(b'%PDF-1.'):
                    text_list = []
                    pdf_reader = PdfReader(io.BytesIO(downloaded))
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()

                        # Split the page text into sentences based on full stops & new lines
                        sentences = page_text.split('.')

                        # Add each sentence to the text list
                        text_list.extend(sentences)

                    return text_list
                else:
                    return downloaded.decode().splitlines()
            else:
                return val.splitlines()

        except StorageException:
            return None

    @staticmethod
    def __download_past_txt_file(user_email, filename):
        try:
            username = user_email.split('@')[0]

            path = DB.construct_path_past(username, filename)
            downloaded = supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).download(path)

            text_lines = []
            for line in downloaded.decode('utf-8').splitlines():
                cleaned_line = line.strip().lstrip("\ufeff")
                text_lines.append(cleaned_line)
            return text_lines

        except StorageException:
            return None

    @staticmethod
    def __download_past_npy_file(user_email, filename):
        username = user_email.split('@')[0]
        try:
            path = DB.construct_path_past(username, filename)
            downloaded = supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).download(path)

            buffer = io.BytesIO(downloaded)
            loaded_data = np.load(buffer, allow_pickle=True)

            return loaded_data

        except StorageException:
            return None

    @staticmethod
    def list_past_assignments(user_email):
        # If there are files in directory, return a list of file names
        # Else, return an empty list

        objects = []
        username = user_email.split('@')[0]
        res = supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).list(username)
        for obj in res:
            if obj['name'] != '.emptyFolderPlaceholder':
                objects.append(obj['name'])
        return objects

    @staticmethod
    def get_past_files(user_email):
        # Old method, likely not used
        # If there are files in directory, return a python list of files (byte streams)
        # Else, return an empty list

        past_assignments = []
        username = user_email.split('@')[0]
        res = supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).list(username)
        for file_object in res:
            try:
                if file_object['name'] != '.emptyFolderPlaceholder':
                    path = DB.construct_path_past(username, file_object['name'])
                    past_assignments.append(supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).download(path))
            except StorageException:
                pass
        return past_assignments

    @staticmethod
    def construct_path_past(username, filename):
        return f'{username}/{filename}'

    @staticmethod
    def construct_path_current(subject_id, assignment_id, user_id):
        return f'{subject_id}/{assignment_id}/{user_id}'

    @staticmethod
    def read_current_assignment(user, filename):
        file_bytes = DB.download_current_assignment(user.subject, user.assignment, user.id)
        if file_bytes is None:
            return None

        if filename[-4:] == '.txt':
            text_lines = []
            for line in file_bytes.decode('utf-8').splitlines():
                cleaned_line = line.strip().lstrip("\ufeff")
                text_lines.append(cleaned_line)
            return text_lines

        elif file_bytes.startswith(b'%PDF-1.'):
            text_list = []
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()

                # Split the page text into sentences based on full stops & new lines
                sentences = page_text.split('.')

                # Add each sentence to the text list
                text_list.extend(sentences)

            return text_list

        else:
            bytesIO = io.BytesIO(file_bytes)
            val = bytesIO.getvalue()

            if type(val) == bytes:
                return file_bytes.decode().splitlines()
            else:
                return val.splitlines()

    @staticmethod
    def download_current_assignment(subject_id, assignment_id, user_id):
        # Will return a byte stream.
        # Essentailly, it returns file.read(): byteStream in python.

        try:
            path = DB.construct_path_current(subject_id, assignment_id, user_id)
            return supabase_sec.storage.from_(CURRENT_ASSIGNMENTS_BUCKET).download(path)
        except StorageException:
            return None

    @staticmethod
    def exists_current_assignment(subject_id, assignment_id, user_id):
        # if the folder is empty, db returns 1 element in list[0] as a placeholder
        res = supabase_sec.storage.from_(CURRENT_ASSIGNMENTS_BUCKET).list(f'{subject_id}/{assignment_id}')
        for obj in res:
            if obj['name'] == user_id:
                return [obj]
        return []

    @staticmethod
    def store_past_style_vector(user, filename, payload):
        temp_userid = user.id.strip('.txt')
        temp_filename = filename.strip('.txt')

        payload.update({'assignment_id': temp_filename, 'user_id': temp_userid})
        supabase_sec.table('SubjectAssignmentUser').upsert(payload).execute()

    @staticmethod
    def store_style_vector(user, payload, prediction):
        temp_userid = user.id.rstrip('.txt')

        payload.update({'assignment_id': user.assignment, 'user_id': temp_userid, 'similarity_score': prediction})
        supabase_sec.table('SubjectAssignmentUser').upsert(payload).execute()


class User:

    def __init__(self, email, subject_id, assignment_id, user_id):
        self.email = email
        self.subject = subject_id
        self.assignment = assignment_id
        self.id = user_id
