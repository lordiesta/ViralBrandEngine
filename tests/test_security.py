import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.auth import require_admin_access, require_api_access


class AccessControlTests(unittest.TestCase):
    def test_api_access_accepts_matching_key(self):
        settings = SimpleNamespace(API_KEY="application-secret", ADMIN_API_KEY="admin-secret")
        with patch("app.auth.get_settings", return_value=settings):
            require_api_access(api_key="application-secret", bearer=None)

    def test_api_access_rejects_missing_key(self):
        settings = SimpleNamespace(API_KEY="application-secret", ADMIN_API_KEY="admin-secret")
        with patch("app.auth.get_settings", return_value=settings), self.assertRaises(HTTPException) as raised:
            require_api_access(api_key=None, bearer=None)
        self.assertEqual(raised.exception.status_code, 401)

    def test_admin_access_is_separate(self):
        settings = SimpleNamespace(API_KEY="application-secret", ADMIN_API_KEY="admin-secret")
        with patch("app.auth.get_settings", return_value=settings):
            require_admin_access(api_key="admin-secret", bearer=None)
            with self.assertRaises(HTTPException) as raised:
                require_admin_access(api_key="application-secret", bearer=None)
        self.assertEqual(raised.exception.status_code, 403)
