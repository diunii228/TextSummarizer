const express = require('express');
const router = express.Router();
const verifyToken = require('../middleware/verifyToken');
const summaryController = require('../controllers/summary.controller');

router.post('/summarize', verifyToken, summaryController.summarize);

module.exports = router;