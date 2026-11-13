// Captures du monde ouvert : node outils/captures.js <url du jeu> <dossier> "<nom:x:y:cap[:vitesse_s]>" ...
const { chromium } = require('playwright');
(async () => {
  const [url, dos, ...vues] = process.argv.slice(2);
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage({ viewport: { width: 1280, height: 760 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(m.text()); });
  for (const v of vues) {
    const [nom, x, y, cap, conduite] = v.split(':');
    const t0 = Date.now();
    await p.goto(`${url}?x=${x}&y=${y}&cap=${cap}${nom.startsWith('demo') ? '&demo_monuments=1' : ''}`, { waitUntil: 'load', timeout: 180000 });
    await p.waitForFunction(() => window.__pret || /pas pu/.test(document.getElementById('etape').textContent), null, { timeout: 240000 });
    const tc = ((Date.now() - t0) / 1000).toFixed(1);
    if (nom.startsWith('mur')) { await p.evaluate(() => { localStorage.removeItem('jeton'); localStorage.setItem('tempsLibre', '298'); }); await p.reload({ waitUntil: 'load' }); await p.waitForFunction(() => window.__pret, null, { timeout: 240000 }); await p.waitForTimeout(8000);
      console.log(nom, 'fenêtre de compte visible :', await p.evaluate(() => !document.getElementById('vCompte').hidden)); }
    if (nom.startsWith('pont')) {   // pont : la voiture sur le plus long tablier des blocs chargés (pontdessus), ou à côté pour le voir (pontcote)
      const ok = await p.evaluate((cote) => { let best = null; for (const b of window.__blocs.values()) if (b.pret) for (const [pts] of (b.M.ponts || [])) { const L = pts.length; if (!best || L > best.L) best = { L, pts, b }; }
        if (!best) return false; const k = Math.floor(best.pts.length / 2), q = best.pts[k], r = best.pts[Math.min(best.pts.length - 1, k + 1)], S = window.__etat, cap = Math.atan2(r[1] - q[1], r[0] - q[0]);
        S.x = q[0] + best.b.ox; S.y = q[1] + best.b.oy; S.cap = S.camCap = cap; if (cote) { S.x += Math.cos(cap + 1.57) * 45 - Math.cos(cap) * 25; S.y += Math.sin(cap + 1.57) * 45 - Math.sin(cap) * 25; S.cap = S.camCap = cap - 1.2; }
        S.v = 0; S.z = S.cz = undefined; return best.pts.length; }, nom.includes('cote'));
      console.log(nom, 'pont trouvé :', ok); await p.waitForTimeout(4000); }
    if (conduite) { await p.keyboard.down('ArrowUp'); await p.waitForTimeout(+conduite * 1000); await p.keyboard.up('ArrowUp'); }
    await p.waitForTimeout(3000);
    const info = await p.evaluate(() => { const S = window.__etat; return S ? { blocs: [...window.__blocs.values()].filter(b => b.pret).length, x: Math.round(S.x), y: Math.round(S.y), z: S.z && +S.z.toFixed(1),
      v: Math.round(Math.abs(S.v) * 3.6), rue: document.getElementById('rue').textContent, ville: document.getElementById('ville').textContent, mem: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null } : { erreur: document.getElementById('etape').textContent }; });
    await p.screenshot({ path: `${dos}/${nom}.png`, timeout: 300000 });
    console.log(nom, 'chargé en', tc, 's', JSON.stringify(info));
  }
  console.log('erreurs et alertes :', JSON.stringify(errs.slice(0, 30)));
  await b.close();
})();
