import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.broll_fetcher import BRollFetcherService, GeminiVideoQuotaError
from app.services.script_generator import ScriptGeneratorService
from app.services.video_editor import VideoEditorService


class VideoGenerationTests(unittest.TestCase):
    def test_placeholder_visual_contains_prompt_subject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "yellow_cat.mp4"
            BRollFetcherService._generate_placeholder_clip(
                output, "yellow cat", duration_secs=1, width=270, height=480, fps=1
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)

    def test_pixabay_queries_keep_one_identifiable_subject(self):
        self.assertEqual(
            BRollFetcherService._clean_pixabay_query(
                "cinematic close-up of yellow cat running away from black dog"
            ),
            "yellow cat running",
        )
        self.assertEqual(
            BRollFetcherService._clean_pixabay_query(
                "running black dog chasing yellow cat"
            ),
            "running black dog",
        )
        self.assertEqual(
            BRollFetcherService._clean_pixabay_query("yellow cat and black dog"),
            "yellow cat",
        )

    def test_pixabay_selects_hit_matching_scene_metadata(self):
        service = BRollFetcherService(pixabay_key="test-key")
        search_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "hits": [
                    {
                        "id": 1,
                        "tags": "city, street, traffic",
                        "videos": {"medium": {"url": "https://example.invalid/city.mp4"}},
                    },
                    {
                        "id": 2,
                        "tags": "running, sports car, road",
                        "videos": {"medium": {"url": "https://example.invalid/car.mp4"}},
                    },
                ]
            },
        )
        download_response = SimpleNamespace(
            content=b"video-bytes",
            raise_for_status=lambda: None,
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.services.broll_fetcher.requests.get",
            side_effect=[search_response, download_response],
        ) as get:
            result = service._fetch_from_pixabay(
                "running sports car",
                Path(temp_dir) / "clip.mp4",
                used_ids=set(),
            )
            self.assertTrue(result.exists())
        self.assertEqual(
            get.call_args_list[1].args[0],
            "https://example.invalid/car.mp4",
        )

    def test_stock_search_rejects_large_cat_as_domestic_cat(self):
        cheetah_score = BRollFetcherService._stock_hit_relevance(
            {"tags": "yellow, cheetah, running, cat"},
            "yellow cat",
            provider="pixabay",
        )
        dog_score = BRollFetcherService._stock_hit_relevance(
            {"tags": "running, black dog, pet"},
            "running black dog",
            provider="pixabay",
        )

        self.assertLess(cheetah_score, 20)
        self.assertGreaterEqual(dog_score, 20)

    def test_local_storyboard_keeps_character_queries_isolated(self):
        with patch.object(
            ScriptGeneratorService, "translate_to_english", return_value="yellow cat escaping a black dog"
        ):
            script = ScriptGeneratorService._mock_response(
                "Sarı kedi siyah köpekten kaçıyor",
                duration=20,
                target_language="tr",
            )

        queries = [scene["visual_search_query"] for scene in script.scenes]
        self.assertEqual(queries, [
            "yellow cat",
            "running yellow cat",
            "running black dog",
            "yellow cat safe",
        ])
        self.assertIn("same black dog", script.scenes[1]["visual_description"])
        self.assertIn("same yellow cat", script.scenes[2]["visual_description"])

    def test_pixabay_does_not_append_topic_anchor_to_scene_query(self):
        service = BRollFetcherService(pixabay_key="test-key")
        hit = {
            "id": 123,
            "tags": "running, black dog, pet",
            "videos": {"medium": {"url": "https://example.invalid/video.mp4"}},
        }
        search_response = SimpleNamespace(status_code=200, json=lambda: {"hits": [hit]})
        download_response = SimpleNamespace(
            content=b"video-bytes",
            status_code=200,
            raise_for_status=lambda: None,
        )
        def mocked_get(url, **kwargs):
            if url == service.PIXABAY_SEARCH_URL:
                return search_response
            return download_response

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.services.broll_fetcher.requests.get",
            side_effect=mocked_get,
        ):
            result = service._fetch_from_pixabay(
                "running black dog chasing yellow cat",
                Path(temp_dir) / "clip.mp4",
                used_ids=set(),
                anchor="yellow cat",
            )
            self.assertTrue(result.exists())

    def test_flow_key_generates_one_continuous_story_and_skips_stock(self):
        service = BRollFetcherService(
            pexels_key="pexels-test",
            pixabay_key="pixabay-test",
            google_flow_key="gemini-test",
            luma_api_key="",
            kling_api_key="",
        )
        scenes = [
            {
                "start_time": 0,
                "end_time": 5,
                "narration_text": "The yellow cat sees danger.",
                "visual_search_query": "yellow cat",
                "visual_description": "Introduce the same yellow cat.",
            },
            {
                "start_time": 5,
                "end_time": 10,
                "narration_text": "The yellow cat runs from the black dog.",
                "visual_search_query": "running yellow cat",
                "visual_description": "The same yellow cat runs from the same black dog.",
            },
        ]

        def write_generated_video(prompt, output_path):
            output_path.write_bytes(b"x" * 11000)
            self.assertIn("same characters throughout", prompt)
            self.assertIn("same black dog", prompt)
            return output_path

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            service, "_generate_with_google_flow", side_effect=write_generated_video
        ) as generate, patch.object(
            service, "_fetch_from_pixabay", side_effect=AssertionError("stock fallback must not run")
        ):
            result = service.fetch_clips(
                ["yellow cat", "running yellow cat"],
                Path(temp_dir),
                topic="Sarı kedi siyah köpekten kaçıyor",
                scenes=scenes,
            )

        self.assertEqual(len(result), 1)
        generate.assert_called_once()

    def test_gemini_interactions_api_downloads_video_data(self):
        video_data = b"valid-mp4-data" * 1000
        encoded = base64.b64encode(video_data).decode("ascii")
        response = SimpleNamespace(
            ok=True,
            json=lambda: {
                "steps": [{
                    "type": "model_output",
                    "content": [{
                        "type": "video",
                        "mime_type": "video/mp4",
                        "data": encoded,
                    }],
                }],
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.services.broll_fetcher.requests.post", return_value=response
        ) as post:
            output = Path(temp_dir) / "generated.mp4"
            BRollFetcherService._generate_with_gemini_video(
                "Continuous story prompt", output, "test-api-key"
            )
            self.assertEqual(output.read_bytes(), video_data)
            self.assertEqual(
                post.call_args.args[0],
                "https://generativelanguage.googleapis.com/v1beta/interactions",
            )
            self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "test-api-key")
            self.assertEqual(post.call_args.kwargs["json"]["response_format"]["aspect_ratio"], "9:16")

    def test_luma_key_takes_priority_over_google_keys(self):
        service = BRollFetcherService(
            gemini_api_key="gemini-test",
            google_flow_key="google-test",
            luma_api_key="luma-test",
            kling_api_key="",
        )

        def write_luma_video(prompt, output_path, api_key):
            self.assertIn("continuous 9:16 vertical video", prompt)
            self.assertEqual(api_key, "luma-test")
            output_path.write_bytes(b"x" * 11000)
            return output_path

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            service, "_generate_with_luma_video", side_effect=write_luma_video
        ) as generate_luma, patch.object(
            service,
            "_generate_with_gemini_video",
            side_effect=AssertionError("Google video generation must be bypassed"),
        ):
            result = service.fetch_clips(
                ["a yellow cat running"],
                Path(temp_dir),
                topic="A yellow cat runs",
            )

        self.assertEqual(len(result), 1)
        generate_luma.assert_called_once()

    def test_kling_key_takes_priority_over_other_ai_video_providers(self):
        service = BRollFetcherService(
            gemini_api_key="gemini-test",
            google_flow_key="google-test",
            luma_api_key="luma-test",
            kling_api_key="kling-test",
        )

        def write_kling_video(prompt, output_path, api_key):
            self.assertIn("continuous 9:16 vertical video", prompt)
            self.assertEqual(api_key, "kling-test")
            output_path.write_bytes(b"x" * 11000)
            return output_path

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            service, "_generate_with_kling_video", side_effect=write_kling_video
        ) as generate_kling, patch.object(
            service,
            "_generate_with_luma_video",
            side_effect=AssertionError("Luma must not run when Kling is configured"),
        ):
            result = service.fetch_clips(
                ["a yellow cat running"],
                Path(temp_dir),
                topic="A yellow cat runs",
            )

        self.assertEqual(len(result), 1)
        generate_kling.assert_called_once()

    def test_kling_generation_submits_polls_and_downloads(self):
        video_data = b"valid-kling-mp4-data" * 1000
        created = SimpleNamespace(
            ok=True,
            json=lambda: {
                "code": 0,
                "data": {"id": "kling-task-123", "status": "submitted"},
            },
        )
        completed = SimpleNamespace(
            ok=True,
            json=lambda: {
                "code": 0,
                "data": [{
                    "id": "kling-task-123",
                    "status": "succeeded",
                    "outputs": [{
                        "type": "video",
                        "url": "https://example.invalid/kling.mp4",
                    }],
                }],
            },
        )
        download = SimpleNamespace(ok=True, content=video_data)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.services.broll_fetcher.requests.post", return_value=created
        ) as post, patch(
            "app.services.broll_fetcher.requests.get",
            side_effect=[completed, download],
        ) as get, patch("app.services.broll_fetcher.time.sleep"):
            output = Path(temp_dir) / "kling.mp4"
            BRollFetcherService._generate_with_kling_video(
                "Continuous story prompt", output, "kling-test-key"
            )

            self.assertEqual(output.read_bytes(), video_data)
            self.assertEqual(
                post.call_args.args[0],
                "https://api-singapore.klingai.com/text-to-video/kling-3.0",
            )
            self.assertEqual(
                post.call_args.kwargs["headers"]["Authorization"],
                "Bearer kling-test-key",
            )
            request_body = post.call_args.kwargs["json"]
            self.assertEqual(request_body["settings"]["duration"], 10)
            self.assertEqual(request_body["settings"]["aspect_ratio"], "9:16")
            self.assertEqual(
                get.call_args_list[0].args[0],
                "https://api-singapore.klingai.com/tasks",
            )
            self.assertEqual(
                get.call_args_list[0].kwargs["params"],
                {"task_ids": "kling-task-123"},
            )
            self.assertEqual(
                get.call_args_list[1].args[0],
                "https://example.invalid/kling.mp4",
            )

    def test_luma_video_generation_submits_polls_and_downloads(self):
        video_data = b"valid-luma-mp4-data" * 1000
        created = SimpleNamespace(
            ok=True,
            json=lambda: {"id": "generation-123", "state": "queued"},
        )
        completed = SimpleNamespace(
            ok=True,
            json=lambda: {
                "id": "generation-123",
                "state": "completed",
                "output": [{"url": "https://example.invalid/generated.mp4"}],
            },
        )
        download = SimpleNamespace(ok=True, content=video_data)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.services.broll_fetcher.requests.post", return_value=created
        ) as post, patch(
            "app.services.broll_fetcher.requests.get",
            side_effect=[completed, download],
        ) as get, patch("app.services.broll_fetcher.time.sleep"):
            output = Path(temp_dir) / "luma.mp4"
            BRollFetcherService._generate_with_luma_video(
                "Continuous story prompt", output, "luma-test-key"
            )
            self.assertEqual(output.read_bytes(), video_data)
            self.assertEqual(
                post.call_args.args[0],
                "https://agents.lumalabs.ai/v1/generations",
            )
            self.assertEqual(
                post.call_args.kwargs["headers"]["Authorization"],
                "Bearer luma-test-key",
            )
            request_body = post.call_args.kwargs["json"]
            self.assertEqual(request_body["model"], "ray-3.2")
            self.assertEqual(request_body["type"], "video")
            self.assertEqual(request_body["aspect_ratio"], "9:16")
            self.assertEqual(request_body["video"]["duration"], "10s")
            self.assertEqual(
                get.call_args_list[0].args[0],
                "https://agents.lumalabs.ai/v1/generations/generation-123",
            )
            self.assertEqual(
                get.call_args_list[1].args[0],
                "https://example.invalid/generated.mp4",
            )

    def test_zero_free_tier_video_quota_reports_billing_action(self):
        response = SimpleNamespace(
            ok=False,
            status_code=429,
            text='{"error":{"message":"Rate limit exceeded (limit: 0 input tokens per minute on Free Tier)"}}',
            json=lambda: {
                "error": {
                    "message": "Rate limit exceeded (limit: 0 input tokens per minute on Free Tier)"
                }
            },
        )

        with patch("app.services.broll_fetcher.requests.post", return_value=response):
            with self.assertRaisesRegex(GeminiVideoQuotaError, "Paid Tier") as raised:
                BRollFetcherService._generate_with_gemini_video(
                    "Continuous story prompt", Path("unused.mp4"), "test-api-key"
                )

        self.assertIn("Yeni API anahtarı tek başına kota sağlamaz", str(raised.exception))

    def test_subtitle_rendering_is_skipped_unless_explicitly_enabled(self):
        class FakeClip:
            duration = 5.0

            def close(self):
                pass

        ffmpeg_result = SimpleNamespace(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "audio.mp3"
            video = Path(temp_dir) / "clip.mp4"
            audio.write_bytes(b"audio")
            video.write_bytes(b"video")
            service = VideoEditorService()
            with patch("moviepy.editor.AudioFileClip", return_value=FakeClip()), patch(
                "moviepy.editor.VideoFileClip", return_value=FakeClip()
            ), patch(
                "app.services.video_editor.subprocess.run", return_value=ffmpeg_result
            ) as run, patch.object(
                service,
                "_build_subtitle_images",
                side_effect=AssertionError("subtitle image generation must be skipped"),
            ) as build_subtitles:
                service._render_with_ffmpeg(
                    [video], audio, "1\n00:00:00,000 --> 00:00:01,000\nTEST",
                    Path(temp_dir) / "without-subtitles.mp4", 5,
                )

        build_subtitles.assert_not_called()
        command = run.call_args.args[0]
        self.assertNotIn("overlay=", command[command.index("-filter_complex") + 1])
        audio_filter = command[command.index("-filter_complex") + 1]
        self.assertIn("apad=whole_dur=5.000000", audio_filter)
        self.assertIn("loudnorm=I=-16:LRA=11:TP=-1.5", audio_filter)


if __name__ == "__main__":
    unittest.main()
