import express from 'express';
import { protect } from '../middleware/auth';
import {
  getAchievements,
  checkAchievements,
  getAchievementProgress,
  getLeaderboard,
} from '../controllers/achievementController';

const router = express.Router();

// All routes are protected
router.use(protect);

router.get('/', getAchievements);
router.post('/check', checkAchievements);
router.get('/progress', getAchievementProgress);
router.get('/leaderboard', getLeaderboard);

export default router;
