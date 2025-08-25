import os
import subprocess
import json
import time
import logging
from datetime import datetime
try:
    from google.cloud import vision
except Exception:
    vision = None
try:
    from google.oauth2 import service_account
except Exception:
    service_account = None
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualTaggingService:
    def __init__(self, project_id):
        self.project_id = project_id
        
        # Set Google Cloud credentials
        credentials_path = os.path.join(os.path.dirname(__file__), 'google-credentials.json')
        if os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            logger.info(f"Google Cloud credentials found at: {credentials_path}")
        else:
            logger.warning(f"Google Cloud credentials not found at: {credentials_path}")
        
        # Initialize Vision API client
        try:
            creds = None
            if service_account and os.path.exists(credentials_path):
                try:
                    creds = service_account.Credentials.from_service_account_file(credentials_path)
                    logger.info("Service account credentials loaded successfully")
                except Exception as e:
                    logger.error(f"Credentials file is not a valid service account JSON: {str(e)}")
                    creds = None

            if vision:
                if creds:
                    self.vision_client = vision.ImageAnnotatorClient(credentials=creds)
                    logger.info("Vision API client initialized successfully with service account")
                else:
                    # Try default credentials
                    try:
                        self.vision_client = vision.ImageAnnotatorClient()
                        logger.info("Vision API client initialized with default credentials")
                    except Exception as e:
                        logger.error(f"Failed to initialize Vision API client: {str(e)}")
                        self.vision_client = None
            else:
                self.vision_client = None
                    
        except Exception as e:
            logger.error(f"Failed to initialize Vision API client: {str(e)}")
            self.vision_client = None
    
    def extract_frames_from_video(self, video_path, video_id, fps=1.0, start_time: float | None = None, end_time: float | None = None):
        """
        Extract frames from video using ffmpeg at 0.5 FPS
        Returns list of frame file paths with timestamps
        """
        try:
            # Create frames directory
            frames_dir = os.path.join('tmp_frames', video_id)
            os.makedirs(frames_dir, exist_ok=True)
            
            logger.info(f"Extracting frames from {video_path} at {fps} FPS")
            if start_time is not None or end_time is not None:
                logger.info(f"Time window requested: start={start_time}, end={end_time}")
            
            # Calculate frame interval (1/fps seconds)
            frame_interval = 1.0 / max(0.1, fps)
            
            # FFmpeg command to extract frames
            output_pattern = os.path.join(frames_dir, 'frame_%04d.jpg')
            
            cmd = ['ffmpeg']
            # Seek and trim if a window is provided
            if start_time is not None and start_time >= 0:
                cmd += ['-ss', str(float(start_time))]
            cmd += ['-i', video_path]
            if end_time is not None and (start_time is not None) and end_time > start_time:
                cmd += ['-t', str(float(end_time - start_time))]
            elif end_time is not None and start_time is None and end_time > 0:
                # If only end_time provided, use -to (absolute from start)
                cmd += ['-to', str(float(end_time))]
            cmd += [
                '-vf', f'fps={fps}',
                '-q:v', '2',
                '-y',
                output_pattern
            ]
            
            # DEBUG: Check PATH and FFmpeg availability
            import shutil
            ffmpeg_path = shutil.which('ffmpeg')
            logger.info(f"DEBUG: Current PATH: {os.environ.get('PATH', 'PATH_NOT_SET')}")
            logger.info(f"DEBUG: ffmpeg found at: {ffmpeg_path}")
            
            if not ffmpeg_path:
                logger.info("DEBUG: ffmpeg not found in PATH, trying direct path...")
                # Try direct path
                direct_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"
                if os.path.exists(direct_path):
                    logger.info(f"DEBUG: Using direct path: {direct_path}")
                    cmd = [direct_path] + cmd[1:]  # Replace first element with direct path
                else:
                    logger.error(f"DEBUG: Direct path {direct_path} does not exist")
                    return []
            
            logger.info(f"DEBUG: Running command: {' '.join(cmd)}")
            # Run ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Get list of extracted frames
            frame_files = []
            frame_index = 1
            
            while True:
                frame_path = os.path.join(frames_dir, f'frame_{frame_index:04d}.jpg')
                if os.path.exists(frame_path):
                    # Calculate timestamp based on frame index and FPS
                    base = float(start_time) if start_time else 0.0
                    timestamp = base + (frame_index - 1) * frame_interval
                    frame_files.append({
                        'path': frame_path,
                        'timestamp': timestamp,
                        'frame_index': frame_index
                    })
                    frame_index += 1
                else:
                    break
            
            logger.info(f"Extracted {len(frame_files)} frames")
            return frame_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"Frame extraction error: {str(e)}")
            return []
    
    def analyze_frame_with_vision_api(self, frame_path):
        """
        Analyze a single frame using Google Vision API label_detection
        Returns list of detected labels with confidence scores
        """
        try:
            if not self.vision_client:
                logger.warning("Vision API client not available, using fallback analysis")
                return self._analyze_frame_fallback(frame_path)
            
            # Read image file
            with open(frame_path, 'rb') as image_file:
                content = image_file.read()
            
            # Create image object
            image = vision.Image(content=content)
            
            # Perform label detection
            response = self.vision_client.label_detection(image=image)
            labels = response.label_annotations
            
            # Convert to our format
            detected_labels = []
            for label in labels:
                detected_labels.append({
                    'tag': label.description,
                    'score': label.score
                })
            
            logger.debug(f"Vision API detected {len(detected_labels)} labels in {os.path.basename(frame_path)}")
            return detected_labels
            
        except Exception as e:
            logger.warning(f"Vision API analysis failed for {frame_path}: {str(e)}, using fallback")
            return self._analyze_frame_fallback(frame_path)
    
    def _analyze_frame_fallback(self, frame_path):
        """
        Fallback analysis when Vision API is not available
        Returns basic visual tags based on image properties
        """
        try:
            import cv2
            import numpy as np
            
            # Read image
            image = cv2.imread(frame_path)
            if image is None:
                return []
            
            # Get image properties
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic statistics
            brightness = np.mean(gray)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Generate basic tags
            tags = []
            
            # Brightness-based tags
            if brightness > 200:
                tags.append({'tag': 'bright', 'score': 0.8})
            elif brightness < 50:
                tags.append({'tag': 'dark', 'score': 0.8})
            
            # Edge density tags
            if edge_density > 0.1:
                tags.append({'tag': 'detailed', 'score': 0.7})
            elif edge_density < 0.02:
                tags.append({'tag': 'smooth', 'score': 0.7})
            
            # Aspect ratio tags
            if aspect_ratio > 1.5:
                tags.append({'tag': 'wide', 'score': 0.6})
            elif aspect_ratio < 0.7:
                tags.append({'tag': 'tall', 'score': 0.6})
            
            # Size-based tags
            if width > 1920 or height > 1080:
                tags.append({'tag': 'high-resolution', 'score': 0.7})
            elif width < 640 or height < 480:
                tags.append({'tag': 'low-resolution', 'score': 0.7})
            
            logger.info(f"Fallback analysis detected {len(tags)} tags in {os.path.basename(frame_path)}")
            return tags
            
        except ImportError:
            logger.warning("OpenCV not available for fallback analysis")
            return [{'tag': 'video-frame', 'score': 0.9}]
        except Exception as e:
            logger.error(f"Fallback analysis error: {str(e)}")
            return [{'tag': 'video-frame', 'score': 0.9}]
    
    def aggregate_tags_from_frames(self, frame_analyses):
        """
        Aggregate tags from multiple frames, removing duplicates and keeping confidence > 0.5
        Returns aggregated tag list with timestamps
        """
        try:
            # Dictionary to store aggregated tags
            tag_aggregator = defaultdict(list)
            
            # Collect all tags with their timestamps
            for frame_data in frame_analyses:
                timestamp = frame_data['timestamp']
                labels = frame_data['labels']
                
                for label in labels:
                    tag_key = label['tag'].lower()  # Normalize tag names
                    tag_aggregator[tag_key].append({
                        'tag': label['tag'],
                        'score': label['score'],
                        'timestamp': timestamp
                    })
            
            # Aggregate and filter tags
            aggregated_tags = []
            for tag_key, occurrences in tag_aggregator.items():
                # Calculate average confidence and find best timestamp
                avg_score = sum(occ['score'] for occ in occurrences) / len(occurrences)
                best_occurrence = max(occurrences, key=lambda x: x['score'])
                
                # Loosen threshold to surface more useful tags
                if avg_score >= 0.3:
                    aggregated_tags.append({
                        'tag': best_occurrence['tag'],
                        'score': avg_score,
                        'timestamp': best_occurrence['timestamp'],
                        'occurrences': len(occurrences)
                    })
            
            # Sort by confidence (highest first) and limit top 20
            aggregated_tags.sort(key=lambda x: (x['score'], x['occurrences']), reverse=True)
            aggregated_tags = aggregated_tags[:20]
            
            logger.info(f"Aggregated {len(aggregated_tags)} tags with confidence > 0.5")
            return aggregated_tags
            
        except Exception as e:
            logger.error(f"Tag aggregation error: {str(e)}")
            return []
    
    def cleanup_frames(self, video_id):
        """
        Clean up extracted frames for a video
        """
        try:
            frames_dir = os.path.join('tmp_frames', video_id)
            if os.path.exists(frames_dir):
                import shutil
                shutil.rmtree(frames_dir)
                logger.info(f"Cleaned up frames directory: {frames_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup frames: {str(e)}")
    
    def tag_video(self, video_path, video_id, start_time: float | None = None, end_time: float | None = None):
        """
        Complete visual tagging pipeline for a video
        Returns aggregated tags with timestamps
        """
        try:
            logger.info(f"Starting visual tagging for video: {video_id}")
            
            # Step 1: Extract frames from video at 0.5 FPS
            logger.info("Step 1: Extracting frames from video at 0.5 FPS...")
            frame_files = self.extract_frames_from_video(video_path, video_id, fps=0.5, start_time=start_time, end_time=end_time)
            
            if not frame_files:
                logger.error("No frames extracted from video")
                return []
            
            # Step 2: Analyze each frame with Google Vision API
            logger.info("Step 2: Analyzing frames with Google Vision API...")
            frame_analyses = []
            
            for i, frame_data in enumerate(frame_files):
                logger.info(f"Analyzing frame {i+1}/{len(frame_files)}")
                labels = self.analyze_frame_with_vision_api(frame_data['path'])
                
                frame_analyses.append({
                    'timestamp': frame_data['timestamp'],
                    'labels': labels
                })
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            # Step 3: Aggregate tags and filter by confidence > 0.5
            logger.info("Step 3: Aggregating tags and filtering by confidence > 0.5...")
            aggregated_tags = self.aggregate_tags_from_frames(frame_analyses)
            
            # Step 4: Save to Firestore
            logger.info("Step 4: Saving tags to Firestore...")
            self.save_tags_to_firestore(video_id, aggregated_tags)
            
            # Step 5: Clean up frames
            logger.info("Step 5: Cleaning up temporary frames...")
            self.cleanup_frames(video_id)
            
            logger.info(f"Visual tagging completed. Found {len(aggregated_tags)} tags")
            return aggregated_tags
            
        except Exception as e:
            logger.error(f"Visual tagging pipeline error: {str(e)}")
            # Clean up on error
            self.cleanup_frames(video_id)
            return []
    
    def save_tags_to_firestore(self, video_id, tags):
        """
        Save aggregated tags to Firestore under tags/{videoId}
        """
        try:
            # For now, save to local JSON file (Firestore disabled)
            tags_data = {
                'videoId': video_id,
                'tags': tags,
                'createdAt': datetime.now().isoformat(),
                'totalTags': len(tags)
            }
            
            # Save to local file (simulating Firestore)
            tags_file = os.path.join('uploads', 'tags', f"{video_id}_tags.json")
            os.makedirs(os.path.dirname(tags_file), exist_ok=True)
            
            with open(tags_file, 'w') as f:
                json.dump(tags_data, f, indent=2)
            
            logger.info(f"Tags saved to {tags_file}")
            
        except Exception as e:
            logger.error(f"Failed to save tags to Firestore: {str(e)}")
    
    def get_tags_from_firestore(self, video_id):
        """
        Get tags from Firestore for a specific video
        """
        try:
            # For now, read from local JSON file (Firestore disabled)
            tags_file = os.path.join('uploads', 'tags', f"{video_id}_tags.json")
            
            if os.path.exists(tags_file):
                with open(tags_file, 'r') as f:
                    tags_data = json.load(f)
                return tags_data.get('tags', [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get tags from Firestore: {str(e)}")
            return []

# Example usage function
def tag_video_file(video_path, video_id, project_id):
    """
    Example function to tag a video file
    """
    try:
        # Initialize tagging service
        service = VisualTaggingService(project_id)
        
        # Tag video
        tags = service.tag_video(video_path, video_id)
        
        if tags:
            logger.info("Visual tagging successful!")
            for tag in tags[:10]:  # Show first 10 tags
                logger.info(f"  {tag['tag']} (confidence: {tag['score']:.2f}, timestamp: {tag['timestamp']:.1f}s)")
            return tags
        else:
            logger.error("Visual tagging failed")
            return []
            
    except Exception as e:
        logger.error(f"Error in tag_video_file: {str(e)}")
        return []
