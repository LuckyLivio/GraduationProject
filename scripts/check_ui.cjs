// Browser integration smoke test. Set PLAYWRIGHT_MODULE to a Playwright module
// path, or install Playwright locally. Requires the localhost server to be up.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'msedge'});
  const page = await browser.newPage({viewport:{width:1440,height:1050},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8765/web/');
  await page.locator('#method-select').waitFor();
  await page.waitForFunction(()=>document.querySelector('#results-body').children.length===6);
  await page.selectOption('#method-select','fixed_16');
  await page.locator('#timeline').fill('12');
  await page.locator('#timeline').dispatchEvent('input');
  fs.mkdirSync(path.resolve('artifacts/demo'),{recursive:true});
  await page.screenshot({path:'artifacts/demo/replay.png',fullPage:true});
  await page.click('#mode-live');
  await page.waitForFunction(()=>document.querySelector('#mode-status').textContent.includes('场景已就绪'));
  await page.selectOption('#method-select','residual_calibrated');
  await page.waitForFunction(()=>document.querySelector('#mode-status').textContent.includes('场景已就绪'));
  await page.locator('#live-damping').fill('1.7');
  await page.locator('#live-damping').dispatchEvent('input');
  await page.click('#reset-button');
  await page.waitForFunction(()=>document.querySelector('#mode-status').textContent.includes('场景已就绪'));
  await page.click('#play-button');
  await page.waitForFunction(()=>Number(document.querySelector('#frame-current').textContent)>=8);
  await page.click('#play-button');
  await page.waitForFunction(()=>document.querySelector('#play-button').textContent.includes('开始'));
  const liveHorizon=await page.locator('#frame-horizon').innerText();
  if(!['5','10','16'].includes(liveHorizon))throw new Error('Missing live horizon');
  await page.screenshot({path:'artifacts/demo/live.png',fullPage:true});
  await page.locator('.lab-grid').screenshot({path:'artifacts/demo/workbench.png'});
  await page.click('#world-canvas',{position:{x:450,y:220}});
  await page.waitForFunction(()=>document.querySelector('#mode-status').textContent.includes('目标已更新'));
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:'artifacts/demo/mobile.png',fullPage:true});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
  if(overflow)throw new Error('Horizontal mobile overflow');
  await browser.close();
  if(errors.length)throw new Error(errors.join('\n'));
  console.log(JSON.stringify({ok:true,liveHorizon,consoleErrors:errors,mobileOverflow:overflow}));
})().catch(e=>{console.error(e);process.exit(1);});
