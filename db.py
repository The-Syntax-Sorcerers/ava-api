from dotenv import load_dotenv
import io
import os

import numpy as np
from storage3.utils import StorageException
from supabase import create_client, Client

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
        else:
            return DB.__download_past_txt_file(user.email, filename)

    @staticmethod
    def __download_past_txt_file(user_email, filename):
        username = user_email.split('@')[0]
        try:
            path = DB.construct_path_past(username, filename)
            temp_file = supabase_sec.storage.from_(PAST_ASSIGNMENTS_BUCKET).download(path)
            text_lines = []
            for line in temp_file.decode('utf-8').splitlines():
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
    def read_current_assignment(user):
        text = DB.download_current_assignment(user.subject, user.assignment, user.id)
        if text is None:
            return None

        text_lines = []
        for line in text.decode('utf-8').splitlines():
            cleaned_line = line.strip().lstrip("\ufeff")
            text_lines.append(cleaned_line)
        return text_lines

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
    def store_style_vector(user, payload):

        payload.update({'subject_id': user.subject, 'assignment_id': user.assignment, 'user_id': user.id,
                        'similarity_score': 0})
        try:
            supabase_sec.table('SubjectAssignmentStudent').upsert(payload).execute()
        except:
            print("Failed to upsert!")




class User:

    def __init__(self, email, subject_id, assignment_id, user_id):
        self.email = email
        self.subject = subject_id
        self.assignment = assignment_id
        self.id = user_id
