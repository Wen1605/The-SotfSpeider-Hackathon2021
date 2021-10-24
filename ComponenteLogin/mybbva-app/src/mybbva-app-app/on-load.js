import {PolymerElement, html} from '@polymer/polymer';
import '@polymer/paper-spinner/paper-spinner.js';

class LoadSpinner extends PolymerElement {
  static get template() {
    return html`
      <paper-spinner active></paper-spinner>
    `;
  }
  static get properties() {
    return {
       buttonVal:{
        type:Boolean,
      },
    };
  }
}
customElements.define('on-load', LoadSpinner);