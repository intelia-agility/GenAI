const crypto = require('crypto');
const KJUR = require('jsrsasign');

// Sign with your SDK Secret
app.get('/generateSignature', (req, res) => {
  const iat = Math.round(new Date().getTime() / 1000) - 30;
  const exp = iat + 60 * 60 * 2;

  const oHeader = { alg: 'HS256', typ: 'JWT' };
  const oPayload = {
    sdkKey: process.env.SDK_KEY,
    mn: 'YOUR_MEETING_ID',
    role: 0,
    iat: iat,
    exp: exp,
    appKey: process.env.SDK_KEY,
    tokenExp: exp
  };

  const sHeader = JSON.stringify(oHeader);
  const sPayload = JSON.stringify(oPayload);
  const signature = KJUR.jws.JWS.sign("HS256", sHeader, sPayload, process.env.SDK_SECRET);
  res.json({ signature });
});
