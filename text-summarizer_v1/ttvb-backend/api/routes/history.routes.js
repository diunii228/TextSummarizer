const express = require('express');
const router = express.Router();
const verifyToken = require('../middleware/verifyToken');
const historyController = require('../controllers/history.controller');

router.get('/history', verifyToken, historyController.getHistory);
router.post('/history', verifyToken, historyController.saveHistory);
router.delete('/history/:id', verifyToken, historyController.deleteHistory);

module.exports = router;